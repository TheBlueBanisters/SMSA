import os
import copy
import torch
from PIL import Image

from torch import nn
from transformers import SiglipProcessor, SiglipModel


import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial


@contextmanager
def timer(hint=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    # print(f"{hint} runtime: {end - start:.4f} s")

INF = 0x7fffffff


class SigLip():

    def __init__(self, device="cuda"):
        self.device = device
        self.model = SiglipModel.from_pretrained(
            "google/siglip-so400m-patch14-384", device_map=self.device)
        self.processor = SiglipProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384")

    def __call__(self, images, texts):
        # with timer("clear_prompt"):
        texts = self.clear_prompt(copy.deepcopy(texts))

        with timer("siglip processer"):
            inputs = self.processor(
                text=texts, images=images, padding="max_length", return_tensors="pt").to(self.model.device)

        # if int(inputs["input_ids"].shape[-1]) > 64:
        #     print(
        #         "[Warning] The input prompt is too long ({}) and will be split and pooling. \n[{}]".format(inputs["input_ids"].shape[-1], texts))

        stride_num = (int(inputs["input_ids"].shape[-1]) + 63) // 64
        stride = (inputs["input_ids"].shape[-1] + stride_num - 1) // stride_num

        input_id_heads, input_id_tails = [], []
        l, r = 0, inputs["input_ids"].shape[-1]
        while l < r:
            input_id_heads.append(inputs["input_ids"][:, l:l + stride])
            l += stride
            if l < r:
                input_id_tails.append(inputs["input_ids"][:, r - stride:r])
                r -= stride

        input_ids = input_id_heads + input_id_tails[::-1]
        input_ids = torch.cat(input_ids)

        with timer("extract embeds"):
            with torch.no_grad():
                with torch.autocast(self.device):
                    outputs = self.model(
                        input_ids, pixel_values=inputs["pixel_values"])
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        return image_embeds, text_embeds.mean(dim=0, keepdim=True)

    def clear_prompt(self, prompt):
        heads = [
            "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.",
            "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
        ]
        tails = [
            "Answer with the option's letter from the given choices directly.",
            "The best answer is:",
            "Answer the question using a single word or phrase.",
            "Only give the best option.\n",
            "Best option: ("
        ]
        for head in heads:
            prompt = prompt.split(head)[-1]
        for tail in tails:
            prompt = prompt.split(tail)[0]
        prompt = prompt.strip()
        return prompt


class MDP3():

    def __init__(self, device="cuda"):
        super().__init__()
        self.n_selection = 8
        self.lamda = 0.2
        self.segment_size = 32 if self.n_selection <= 32 else 128
        self.condition_size = 1

        self.kernel = MultiGaussianKernel(
            alphas=[2**k for k in list(range(-3, 2))])
        self.vlm = SigLip(device)

    def __call__(self, frames, prompt, sample=None):
        with timer("** TOTAL Frame Selection"):
            input_frames = copy.deepcopy(frames)
            with timer("Read Image"):
                if isinstance(frames[0], str):
                    frames = [Image.open(frame_path) for frame_path in frames]
            with timer("** VLMs Process & Extract"):
                image_embeds, text_embeds = self.vlm(frames, prompt)
            with timer("Select Frames"):
                with torch.no_grad():
                    selected_idx = self._select_frames_fast(
                        image_embeds, text_embeds)
            # with timer("Select Frames Base"):
            #     with torch.no_grad():
            #         selected_idx_base = self._select_frames(image_embeds, text_embeds)
            # print(selected_idx_base)
            # print(selected_idx)
            # assert len(selected_idx) == len(selected_idx_base)
            # for a, b in zip(selected_idx, selected_idx_base):
            #     assert a == b
            return [input_frames[idx] for idx in selected_idx]

    def cal_obj(self, selected_images_embeds, text_embed):
        # with timer("cal kernel_matrix"):
        kernel_matrix = self.kernel(
            torch.cat([text_embed, selected_images_embeds]))
        r, S_matrix = kernel_matrix[0:1, 1:], kernel_matrix[1:, 1:]
        ret_score = (1. / self.lamda * 2 * torch.log(r).sum()) + \
            torch.linalg.slogdet(S_matrix).logabsdet
        return ret_score

    def _select_frames(self, image_embeds, text_embeds):
        # initializing dynamic programing
        N_image = len(image_embeds)
        segment_num = (N_image + self.segment_size - 1) // self.segment_size
        dp = [[0.] + [-INF] * self.n_selection for _ in range(segment_num + 1)]
        trace = [[[] for _ in range(self.n_selection + 1)]
                 for _ in range(segment_num + 1)]

        for seg_idx in range(1, segment_num + 1):
            for selected_num in range(1, min(self.n_selection, seg_idx * self.segment_size) + 1):
                # with timer(f"seqdpp_select_{seg_idx}_{selected_num}"):
                for to_select_num in range(0, min(selected_num, self.segment_size) + 1):
                    cur_score, cur_trace = self.seqdpp_select(
                        text_embeds=text_embeds,
                        image_embeds=image_embeds,
                        conditional_index=trace[seg_idx - 1][selected_num - to_select_num][
                            -min(self.condition_size, len(trace[seg_idx - 1][selected_num - to_select_num])):],
                        candidate_index=range(
                            (seg_idx - 1) * self.segment_size, seg_idx * self.segment_size),
                        to_select_num=to_select_num
                    )
                    cur_score = dp[seg_idx - 1][selected_num -
                                                to_select_num] + cur_score
                    cur_trace = trace[
                        seg_idx - 1][selected_num - to_select_num] + cur_trace
                    if cur_score > dp[seg_idx][selected_num]:
                        dp[seg_idx][selected_num] = cur_score
                        trace[seg_idx][selected_num] = cur_trace
        return trace[segment_num][self.n_selection]

    def _select_frames_fast(self, image_embeds, text_embeds):
        # initializing dynamic programing
        N_image = len(image_embeds)
        segment_num = (N_image + self.segment_size - 1) // self.segment_size
        dp = [[0.] + [-INF] * self.n_selection for _ in range(segment_num + 1)]
        trace = [[[] for _ in range(self.n_selection + 1)]
                 for _ in range(segment_num + 1)]

        for seg_idx in range(1, segment_num + 1):
            candidate_index = range(
                (seg_idx - 1) * self.segment_size, seg_idx * self.segment_size)
            candidate_embeds = [image_embeds[i] for i in candidate_index]
            sim_matrix = self.kernel(torch.stack(candidate_embeds))

            for start_selected_num in range(0, min(self.n_selection, (seg_idx - 1) * self.segment_size) + 1):
                conditional_index = trace[seg_idx - 1][start_selected_num][
                    -min(self.condition_size, len(trace[seg_idx - 1][start_selected_num])):]
                offset = len(conditional_index)
                additional_embeds = [text_embeds[
                    0].reshape(-1)] + [image_embeds[i] for i in conditional_index]
                additional = self.kernel(
                    torch.stack(additional_embeds),
                    torch.stack(additional_embeds + candidate_embeds)
                )
                total_matrix = torch.cat([
                    additional,  # [add, 32+add]
                    torch.cat([
                        additional[:, -len(sim_matrix):].T,  # [32, add]
                        sim_matrix  # [32, 32]
                    ], dim=1)  # [32, add + 32]
                ], dim=0)  # [add+32, add+32]
                # I = torch.diag(torch.tensor([0.] * offset + [1.] * len(candidate_index), device=total_matrix.device))
                # v = -torch.linalg.slogdet(total_matrix[1:,1:] + I).logabsdet

                max_selection = min(self.n_selection -
                                    start_selected_num, self.segment_size)
                # for to_select_num in range(0, max_selection + 1):
                #     with timer(f"seqdpp_select_{seg_idx}_{start_selected_num}_{to_select_num}"):
                #         cur_score, cur_trace = self.seqdpp_select_fast(
                #             total_matrix, offset, to_select_num)
                #         cur_trace = [i + int((seg_idx - 1) * self.segment_size) for i in cur_trace]

                # with timer(f"seqdpp_select_{seg_idx}_{start_selected_num}"):
                cur_scores, cur_traces = self.seqdpp_select_super_fast(
                    total_matrix, offset, max_selection)
                # print(cur_scores, cur_traces)
                for to_select_num, (cur_score, cur_trace) in enumerate(zip(cur_scores, cur_traces)):
                    # cur_score_base, cur_trace_base = self.seqdpp_select_fast(total_matrix, offset, to_select_num)
                    # cur_trace_base = [i + int((seg_idx - 1) * self.segment_size) for i in cur_trace_base]

                    cur_trace = [i + int((seg_idx - 1) * self.segment_size)
                                 for i in cur_trace]

                    # print("Base", to_select_num, cur_score_base, cur_trace_base)
                    # print("Superfast", to_select_num, cur_score, cur_trace)
                    cur_score = dp[seg_idx - 1][start_selected_num] + cur_score
                    cur_trace = trace[
                        seg_idx - 1][start_selected_num] + cur_trace

                    if cur_score > dp[seg_idx][start_selected_num + to_select_num]:
                        dp[seg_idx][start_selected_num + to_select_num] = cur_score
                        trace[seg_idx][start_selected_num +
                                       to_select_num] = cur_trace
        return trace[segment_num][self.n_selection]

    def seqdpp_select(self, text_embeds, image_embeds, conditional_index, candidate_index, to_select_num):
        if to_select_num == 0:
            return 0.0, []
        conditional_embeds = [image_embeds[i] for i in conditional_index]
        cur_trace = []
        U_matrix = self.kernel(torch.stack(
            conditional_embeds + [image_embeds[i] for i in candidate_index]))
        I = torch.diag(
            torch.tensor([0.] * len(conditional_index) + [1.] *
                         len(candidate_index), device=U_matrix.device)
        )
        obj_values = -torch.linalg.slogdet(U_matrix + I).logabsdet
        while len(cur_trace) < to_select_num:
            max_obj_gain = -INF
            cur_selected_idx = -1
            for i in candidate_index:
                if i in cur_trace:
                    continue
                cur_obj = self.cal_obj(
                    selected_images_embeds=torch.stack(
                        conditional_embeds + [image_embeds[j] for j in cur_trace + [i]]),
                    text_embed=text_embeds[0].reshape(1, -1)
                )
                cur_obj_gain = cur_obj - obj_values
                if cur_obj_gain > max_obj_gain:
                    max_obj_gain = cur_obj_gain
                    cur_selected_idx = i
            cur_trace.append(cur_selected_idx)
            obj_values += max_obj_gain
        cur_trace = sorted(cur_trace)
        return obj_values if len(cur_trace) > 0 else 0.0, cur_trace

    def seqdpp_select_fast(self, total_matrix, offset, to_select_num):
        if to_select_num == 0:
            return 0.0, []
        cur_trace = []
        obj_values = 0.0
        r, S_matrix = total_matrix[0:1, 1:], total_matrix[1:, 1:]
        candidate_index = range(len(S_matrix) - offset)

        while len(cur_trace) < to_select_num:
            max_obj_gain = -INF
            cur_selected_idx = -1
            for i in candidate_index:
                if i in cur_trace:
                    continue
                selected_idx = list(range(offset)) + \
                    [j + offset for j in cur_trace + [i]]
                cur_S_matrix = S_matrix[selected_idx][:, selected_idx]
                cur_obj = (1. / self.lamda * 2 * torch.log(
                    r[:, selected_idx]).sum()) + torch.linalg.slogdet(cur_S_matrix).logabsdet
                cur_obj_gain = cur_obj - obj_values
                if cur_obj_gain > max_obj_gain:
                    max_obj_gain = cur_obj_gain
                    cur_selected_idx = i
            cur_trace.append(cur_selected_idx)
            obj_values += max_obj_gain
        cur_trace = sorted(cur_trace)
        return obj_values if len(cur_trace) > 0 else 0.0, cur_trace

    def seqdpp_select_super_fast(self, total_matrix, offset, to_select_num):
        if to_select_num == 0:
            return [0.0], [[]]
        cur_trace = []
        ret_scores = [0.0]
        r, S_matrix = total_matrix[0:1, 1:], total_matrix[1:, 1:]
        candidate_index = list(range(len(S_matrix) - offset))

        conditional_idx = list(range(offset))
        L = None
        if len(conditional_idx) > 0:
            L = torch.linalg.cholesky(
                S_matrix[conditional_idx][:, conditional_idx])

        while len(cur_trace) < to_select_num:
            max_obj = -INF
            cur_selected_idx = -1
            better_L = None
            for i in candidate_index:
                if i in cur_trace:
                    continue
                cur_idx = i + offset
                selected_idx = conditional_idx + \
                    [j + offset for j in cur_trace] + [cur_idx]
                if L is None:
                    cur_sim_v = S_matrix[selected_idx][:, selected_idx]
                    cur_L = torch.sqrt(cur_sim_v).reshape(1, 1)
                    logdet = cur_sim_v.clone().log()
                else:
                    cur_sim_v = S_matrix[cur_idx:cur_idx + 1][:, selected_idx]
                    cur_L, logdet = self.cholesky_update_determinant(
                        L, cur_sim_v)
                cur_obj = 1. / self.lamda * 2 * \
                    torch.log(r[:, selected_idx]).sum() + logdet

                if cur_obj > max_obj or cur_selected_idx == -1:
                    max_obj = cur_obj
                    cur_selected_idx = i
                    better_L = cur_L
            ret_scores.append(max_obj.clone())
            cur_trace.append(cur_selected_idx)
            L = better_L
        ret_traces = [sorted(cur_trace[:j]) for j in range(len(cur_trace) + 1)]
        return ret_scores, ret_traces

    def cholesky_update_determinant(self, L, v):
        n = L.shape[0]
        v = v.view(-1, 1)
        v_projected = torch.linalg.solve_triangular(L, v[:n], upper=False)

        new_diag_element = torch.sqrt(torch.abs(v[-1] - v_projected.T @ v_projected))

        new_row = torch.cat((v_projected.flatten(), new_diag_element.view(1)))
        new_L = torch.zeros((n + 1, n + 1), dtype=L.dtype, device=L.device)
        new_L[:n, :n] = L
        new_L[n, :n] = new_row[:-1]
        new_L[n, n] = new_diag_element

        new_diag = torch.diag(new_L)
        # new_det = torch.prod(new_diag) ** 2
        new_logdet = 2 * torch.log(new_diag).sum()

        return new_L, new_logdet


class GaussianKernel(nn.Module):

    def __init__(self, alpha=1.):
        super(GaussianKernel, self).__init__()
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(1) - X.unsqueeze(0)) ** 2).sum(2)
        return torch.exp(-l2_distance_square / (2 * self.alpha))


class MultiGaussianKernel(nn.Module):

    def __init__(self, alphas=[2**k for k in list(range(-3, 2))]):
        super(MultiGaussianKernel, self).__init__()
        self.alphas = alphas

    def forward(self, X: torch.Tensor, Y: torch.tensor = None) -> torch.Tensor:
        Y = X.unsqueeze(0) if Y is None else Y.unsqueeze(0)
        X = X.unsqueeze(1)
        l2_distance_square = ((X - Y) ** 2).sum(2)

        return sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in self.alphas])
