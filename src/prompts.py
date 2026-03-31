"""
Prompt templates for TOPreward experiments.

ORIGINAL is the exact template from the TOPreward paper.
VARIANTS are 9 paraphrases for the prompt ensemble experiment.
All templates end with "The answer is:" so the next token is True/False.
"""

ORIGINAL = (
    "The above images show a robot manipulation trajectory that completes "
    "the following task: {task}. "
    "Decide whether the above statement is True or not. The answer is:"
)

VARIANTS = [
    # 0 — original
    ORIGINAL,
    # 1 — agent-centric phrasing
    (
        "The above images show a robot attempting the task: {task}. "
        "Based on the images, has the robot successfully completed this task? "
        "True or False. The answer is:"
    ),
    # 2 — question form
    (
        "Looking at this sequence of robot images, the task '{task}' "
        "has been completed. Is this statement True or False? The answer is:"
    ),
    # 3 — completion-focused
    (
        "The robot in these images is performing: {task}. "
        "The task is now finished and the goal has been achieved. "
        "True or False? The answer is:"
    ),
    # 4 — outcome-focused
    (
        "These images depict a robot trajectory. The intended task was: {task}. "
        "The desired outcome has been reached. True or False. The answer is:"
    ),
    # 5 — minimal
    (
        "Robot task: {task}. "
        "The above images show this task being completed. True or False? The answer is:"
    ),
    # 6 — progress-framing
    (
        "The above robot trajectory shows progress on: {task}. "
        "The task has been fully completed by the end of the sequence. "
        "True or False? The answer is:"
    ),
    # 7 — confirmation request
    (
        "Please confirm: the robot manipulation shown above successfully "
        "completes the task '{task}'. True or False? The answer is:"
    ),
    # 8 — goal-state framing
    (
        "The above images show a robot. The goal is: {task}. "
        "The goal state has been achieved in these images. "
        "Answer True or False. The answer is:"
    ),
    # 9 — negative framing flipped
    (
        "Examining the robot images above, there is no failure in completing "
        "the task: {task}. The task succeeded. True or False? The answer is:"
    ),
]


def build_prompt(template: str, task: str) -> str:
    return template.format(task=task)
