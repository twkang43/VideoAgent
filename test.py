import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI

from binary_search import retrieve_best_frame
from utils_clip import frame_retrieval_seg_ego
from utils_general import get_from_cache, save_to_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("egoschema_subset.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


client = OpenAI()


def parse_json(text):
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        print("No valid JSON found in the text.")
        return None


def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1


def parse_text_find_confidence(text):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1


def get_llm_response(
    system_prompt, prompt, json_format=True, model="gpt-4-1106-preview"
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])
    logger.info(messages)
    # cached_value = get_from_cache(key)
    # if cached_value is not None:
    #     logger.info("Cache Hit")
    #     logger.info(cached_value)
    #     return cached_value

    print("Not hit cache", key)
    input()

    for _ in range(3):
        try:
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
            response = completion.choices[0].message.content
            logger.info(response)
            save_to_cache(key, response)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"


def generate_final_answer(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think carefully and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def generate_description_step(question, caption, num_frames, segment_des):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
        ],
        "best_segment": 
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"}
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def self_eval(previous_prompt, answer):
    confidence_format = {"confidence": "xxx"}
    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def ask_gpt_caption(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of five uniformly sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def ask_gpt_caption_step(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def read_caption(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption

######################################################################################################
def ask_gpt_select_best_frame(partial_caps, formatted_question):
    return partial_caps

def select_best_sub_segment(captions, question):
    caption_i, caption_k, caption_j = captions
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }
    prompt = f"""
    Given the following descriptions of sampled frames in the video:
    {[caption_i, caption_k, caption_j]}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response

def retrieve_best_frame(caps, duration, formatted_question):
    i, j = duration
    
    if j - i + 1 <= 3:
        return ask_gpt_select_best_frame(caps[i:j+1], formatted_question)
    
    k = (i + j) // 2
    
    caption_k = caps[k]
    caption_i = caps[i]
    caption_j = caps[j]
    
    description_left = generate_description_for_segment(caption_i, caption_k)
    description_right = generate_description_for_segment(caption_k, caption_j)
    
    candidate_descriptions = select_best_sub_segment()
    
    if select_best_segement(description_left, description_right):
        return retrieve_best_frame(caps, [caption_i, caption_k], formatted_question)
    else:
        return retrieve_best_frame(caps, [caption_k, caption_j], formatted_question)
######################################################################################################

def run_one_question(video_id, ann, caps, logs):
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    num_frames = len(caps)

    ### Step 1 ###
    sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
    sampled_caps = read_caption(caps, sample_idx)
    # previous_prompt, answer_str = ask_gpt_caption(
    #     formatted_question, sampled_caps, num_frames
    # )
    # answer = parse_text_find_number(answer_str)
    # confidence_str = self_eval(previous_prompt, answer_str)
    # confidence = parse_text_find_confidence(confidence_str)
    
    confidence = 1

    ### Step 2 ###
    if confidence < 4:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }

            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            logger.info(f"parsed_candidate_descriptions : {parsed_candiate_descriptions}")
            # frame_idx = frame_retrieval_seg_ego(
            #     parsed_candiate_descriptions["best_segment"], video_id, sample_idx
            # )
            
            best_duration = list(map(int, parsed_candiate_descriptions["best_segment"]["duration"].split('-')))
            logger.info(f"best_duration : {best_duration}")
            frame_idx = retrieve_best_frame(
                caps, best_duration, formatted_question
            )
            logger.info(f"Step 2: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))

            # sampled_caps = read_caption(caps, sample_idx)
            # previous_prompt, answer_str = ask_gpt_caption_step(
            #     formatted_question, sampled_caps, num_frames
            # )
            # answer = parse_text_find_number(answer_str)
            # confidence_str = self_eval(previous_prompt, answer_str)
            # confidence = parse_text_find_confidence(confidence_str)
        except Exception as e:
            logger.error(f"Step 2 Error: {e}")
            # answer_str = generate_final_answer(
            #     formatted_question, sampled_caps, num_frames
            # )
            # answer = parse_text_find_number(answer_str)

    if answer == -1:
        logger.info("Answer Index Not Found!")
        answer = random.randint(0, 4)

    logger.info(f"Finished video: {video_id}/{answer}/{ann['truth']}")

    label = int(ann["truth"])
    corr = int(label == answer)
    count_frame = len(sample_idx)

    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": corr,
        "count_frame": count_frame,
    }


def main():
    # if running full set, change subset to fullset
    input_ann_file = "subset_anno.json"
    all_cap_file = "lavila_subset.json"
    json_file_name = "egoschema_subset.json"

    anns = json.load(open(input_ann_file, "r"))
    all_caps = json.load(open(all_cap_file, "r"))
    logs = {}

    tasks = [
        (video_id, anns[video_id], all_caps[video_id], logs)
        for video_id in list(anns.keys())[2:3]
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(lambda p: run_one_question(*p), tasks)

    json.dump(logs, open(json_file_name, "w"))


if __name__ == "__main__":
    main()
