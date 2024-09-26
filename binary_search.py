from typing import List

import numpy as np

from test import read_caption, get_llm_response

def ask_gpt_select_best_frame(partial_caps, formatted_question):
    return partial_caps

def generate_description_for_segment(caption_left, caption_right, question):
    caption = read_caption(caps, [caption_left, caption_right])
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
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

def select_best_segement(description_left, description_right):
    pass

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
    
    if select_best_segement(description_left, description_right):
        return retrieve_best_frame(caps, [caption_i, caption_k], formatted_question)
    else:
        return retrieve_best_frame(caps, [caption_k, caption_j], formatted_question)