import json
from enum import Enum
import time
import base64

import google.generativeai as genai
import os
import mimetypes
from graph_eqa.utils.data_utils import get_latest_image

if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
else:
    print('Gemini token has not been set up yet!')

# Choose a Gemini model.
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-pro-preview-03-25")
# gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash-preview-04-17")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options):
    
    frontier_step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'explanation_frontier': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain reasoning for choosing this frontier to explore by referencing list of objects (<id> and <name>) connected to that frontier node via a link (refer to scene graph)."
            ),
            'frontier_id': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.value for member in frontier_node_list]
            ),
        },
        required=['explanation_frontier', 'frontier_id']
    )

    object_step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'explanation_room': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain very briefly reasoning for selecting this room."
            ),
            'room_id': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.name for member in room_node_list],
                description="Choose the room which contains the object you want to goto."
            ),
            'room_name': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Refer to the the scene graph to output the room_name corresponding to the selected room_id"         
            ),
            'explanation_obj': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain very briefly reasoning for selecting this object in the selected room."
            ),
            # 'explanation_region': genai.protos.Schema(
            #     type=genai.protos.Type.STRING,
            #     description="Explain very briefly reasoning for selecting this region."
            # ),
            
            # 'region_id': genai.protos.Schema(
            #     type=genai.protos.Type.STRING,
            #     enum=[member.name for member in region_node_list],
            #     description="Only select from region nodes connected to the room node (in the room)."
            # ),
            'object_id': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.name for member in object_node_list],
                description="Only select from objects within the room chosen."
            ),
            'object_name': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Refer to the the scene graph to output the object_name corresponding to the selected object_id" 
            )
        },
        # required=['explanation_room', 'explanation_region', 'explanation_obj', 'room_id', 'region_id', 'object_id']
        required=['explanation_room', 'explanation_obj', 'room_id', 'room_name', 'object_id', 'object_name']

    )

    answer = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'explanation_ans': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Select the correct answer from the options."
            ),
            'answer': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.name for member in Answer_options]
            ),
            'value': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.value for member in Answer_options]
            ),
            'explanation_conf': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain the reasoning behind the confidence level of your answer."
            ),
            'confidence_level': genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="Rate your level of confidence. Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain that you can answer the question. This value represents your confidence in answering the question correctly, and not confidence pertaining to choosing the next actions."
            ),
            'is_confident': genai.protos.Schema(
                type=genai.protos.Type.BOOLEAN,
                description="Choose TRUE, if you are very confident about answering the question correctly.  Very IMPORTANT: Only answer TRUE when you have a visual confirmation (from the image) as well as from the scene graph that your answer is correct. Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better. Choose 'FALSE', if you are uncertain of the answer. Do not be overconfident. Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly."
            )
        },
        required=['explanation_ans', 'answer', 'value', 'explanation_conf', 'confidence_level', 'is_confident']
    )

    image_description = genai.protos.Schema(
        type = genai.protos.Type.STRING,
        description="Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question or select future actions."
    )

    scene_graph_description = genai.protos.Schema(
        type = genai.protos.Type.STRING,
        description="Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions."
    )

    question_type = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        enum=["Identification", "Counting", "Existence", "State", "Location"],
        description="Use this to describe the type of question you are being asked."
    )

    step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'Goto_frontier_node_step': frontier_step,  # Ensure these are Schema objects
            'Goto_object_node_step': object_step
        },
        description="Choose only one of 'Goto_frontier_node_step', 'Goto_object_node_step'."
    )

    steps = genai.protos.Schema(
        type = genai.protos.Type.ARRAY,
        items = step,
        min_items = 1)

    response_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties = {
            'steps': steps,
            'image_description': image_description,
            'scene_graph_description': scene_graph_description,
            'question_type': question_type,
            'answer': answer
        },
        required=['steps', 'image_description', 'scene_graph_description', 'question_type', 'answer']
    )

    return response_schema

class VLMPlannerEQAGemini:
    def __init__(self, cfg, sg_sim, question, pred_candidates, choices, answer, output_path):
        
        self._question, self.choices, self.vlm_pred_candidates = question, choices, pred_candidates
        self._answer = answer
        self._output_path = output_path
        self._vlm_type = cfg.name
        self._use_image = cfg.use_image

        self._example_plan = '' #TODO(saumya)
        self._history = ''
        self.full_plan = ''
        self._t = 0
        self._add_history = cfg.add_history

        self._outputs_to_save = [f'Question: {self._question}. \n Answer: {self._answer} \n']
        self.sg_sim = sg_sim
        #self.temp = cfg.temp

    @property
    def t(self):
        return self._t
    
    def get_actions(self): 
        object_node_list = Enum('object_node_list', {id: name for id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names)}, type=str)
        
        if len(self.sg_sim.frontier_node_ids)> 0:
            frontier_node_list = Enum('frontier_node_list', {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str)
        else:
            frontier_node_list = Enum('frontier_node_list', {'frontier_0': 'Do not choose this option. No more frontiers left.'}, type=str)
        room_node_list = Enum('room_node_list', {id: name for id, name in zip(self.sg_sim.room_node_ids, self.sg_sim.room_node_names)}, type=str)
        region_node_list = Enum('region_node_list', {ac: ac for ac in self.sg_sim.region_node_ids}, type=str)
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(self.vlm_pred_candidates, self.choices)}, type=str)
        return frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options
    
    
    @property
    def agent_role_prompt(self):
        scene_graph_desc = "A scene graph represents an indoor environment in a hierarchical tree structure consisting of nodes and edges/links. There are six types of nodes: building, rooms, visited areas, frontiers, objects, and agent in the environemnt. \n \
            The tree structure is as follows: At the highest level 5 is a 'building' node. \n \
            At level 4 are room nodes. There are links connecting the building node to each room node. \n \
            At the lower level 3, are region and frontier nodes. 'region' node represent region of room that is already explored. Frontier nodes represent areas that are at the boundary of visited and unexplored areas. There are links from room nodes to corresponding region and frontier nodes depicted which room they are located in. \n \
            At the lowest level 2 are object nodes and agent nodes. There is an edge from region node to each object node depicting which visited area of which room the object is located in. \
            There are also links between frontier nodes and objects nodes, depicting the objects in the vicinity of a frontier node. \n \
            Finally the agent node is where you are located in the environment. There is an edge between a region node and the agent node, depicting which visited area of which room the agent is located in."
        current_state_des = "'CURRENT STATE' will give you the exact location of the agent in the scene graph by giving you the agent node id, location, room_id and room name. "
        
        if self._use_image:
            current_state_des += " Additionally, you will also be given the current view of the agent as an image. "
        
        prompt = f'''You are an excellent hierarchical graph planning agent. 
            Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.
            As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  
            {scene_graph_desc}. {current_state_des} 
            Given the current state information, try to answer the question. Explain the reasoning for selecting the answer.
            Finally, report whether you are confident in answering the question. 
            Explain the reasoning behind the confidence level of your answer. Rate your level of confidence. 
            Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain.
            Do not use just commensense knowledge to decide confidence. 
            Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better. 
            Choose 'FALSE', if you are uncertain of the answer and should explore more to ground your answer in the current envioronment. 
            Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly.
            If you are unable to answer the question with high confidence, and need more information to answer the question, then you can take two kinds of steps in the environment: Goto_object_node_step or Goto_frontier_node_step 
            You also have to choose the next action, one which will enable you to answer the question better. 
            Goto_object_node_step: Navigates near a certain object in the scene graph. Choose this action to get a good view of the region aroung this object, if you think going near this object will help you answer the question better.
            Important to note, the scene contains incomplete information about the environment (objects maybe missing, relationships might be unclear), so it is useful to go near relevant objects to get a better view to answer the question. 
            Use a scene graph as an imperfect guide to lead you to relevant regions to inspect.
            Choose the object in a hierarchical manner by first reasoning about which room you should goto to best answer the question, and then choose the specific object. \n
            Goto_frontier_node_step: If you think that using action "Goto_object_node_step" is not useful, in other words, if you think that going near any of the object nodes in the current scene graph will not provide you with any useful information to answer the question better, then choose this action.
            This action will navigate you to a frontier (unexplored) region of the environment and will provide you information about new objects/rooms not yet in the scene graph. It will expand the scene graph. 
            Choose this frontier based on the objects connected this frontier, in other words, Goto the frontier near which you see objects that are useful for answering the question or seem useful as a good exploration direction. Explain reasoning for choosing this frontier, by listing the list of objects (<id> and <name>) connected to this frontier node via a link (refer to scene graph) \n \
            
            While choosing either of the above actions, play close attention to 'HISTORY' especially the previous 'Action's to see if you have taken the same action at previous timesteps. 
            Avoid taking the same actions you have taken before.
            Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question or select future actions.
            Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions.
            '''

        prompt_no_image = f'''You are an excellent hierarchical graph planning agent. 
            Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.
            As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  
            {scene_graph_desc}. {current_state_des} 
            Given the current state information, try to answer the question. Explain the reasoning for selecting the answer.
            Finally, report whether you are confident in answering the question. 
            Explain the reasoning behind the confidence level of your answer. Rate your level of confidence. 
            Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain.
            Do not use just commensense knowledge to decide confidence. 
            Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better. 
            Choose 'FALSE', if you are uncertain of the answer and should explore more to ground your answer in the current envioronment. 
            Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly.
            If you are unable to answer the question with high confidence, and need more information to answer the question, then you can take two kinds of steps in the environment: Goto_object_node_step or Goto_frontier_node_step 
            You also have to choose the next action, one which will enable you to answer the question better. 
            Goto_object_node_step: Navigates near a certain object in the scene graph. Choose this action to go to the region aroung this object, if you think going near this object will help you answer the question better.
            Choose the object in a hierarchical manner by first reasoning about which room you should goto to best answer the question, and then choose the specific object. \n
            Goto_frontier_node_step: If you think that using action "Goto_object_node_step" is not useful, in other words, if you think that going near any of the object nodes in the current scene graph will not provide you with any useful information to answer the question better, then choose this action.
            This action will navigate you to a frontier (unexplored) region of the environment and will provide you information about new objects/rooms not yet in the scene graph. It will expand the scene graph. 
            Choose this frontier based on the objects connected this frontier, in other words, Goto the frontier near which there are objects useful for answering the question or seem useful as a good exploration direction. Explain reasoning for choosing this frontier, by listing the list of objects (<id> and <name>) connected to this frontier node via a link (refer to scene graph) \n \
            
            While choosing either of the above actions, play close attention to 'HISTORY' especially the previous 'Action's to see if you have taken the same action at previous timesteps. 
            Avoid taking the same actions you have taken before.
            Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions.
            '''
        
        if self._use_image:
            return prompt
        else:
            return prompt_no_image

    def get_current_state_prompt(self, scene_graph, agent_state):
        # TODO(saumya): Include history
        prompt = f"At t = {self.t}: \n \
            CURRENT AGENT STATE: {agent_state}. \n \
            SCENE GRAPH: {scene_graph}. \n "

        if self._add_history:
            prompt += f"HISTORY: {self._history}"

        return prompt

    def update_history(self, agent_state, step, question_type):
        if (step['step_type'] == 'Goto_object_node_step'):
            action = f"Goto object_id: {step['choice']} object name: {step['value']}"
        elif step['step_type'] == 'Goto_frontier_node_step':
            action = f"Goto frontier_id: {step['choice']}"
        else:
            action = f"Answer: {step['choice']}: {step['value']}.  Confident: {step['is_confident']}, Confidence level:{step['confidence_level']}"
        
        last_step = f'''
            [Agent state(t={self.t}): {agent_state}, 
            Action(t={self.t}): {action},
            Question Type: {question_type} 
        '''
        self._history += last_step
    
    def get_gemini_output(self, current_state_prompt):
        # TODO(blake):
        messages=[
            {"role": "model", "parts": [{"text": f"AGENT ROLE: {self.agent_role_prompt}"}]},
            {"role": "model", "parts": [{"text": f"QUESTION: {self._question}"}]},
            {"role": "user", "parts": [{"text": f"CURRENT STATE: {current_state_prompt}."}]},
        ]
        print(messages)
        
        if self._use_image:
            image_path = get_latest_image(self._output_path)
            base64_image = encode_image(image_path) 
            mime_type = mimetypes.guess_type(image_path)[0]
            messages.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question."
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            )

        frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options = self.get_actions()

        succ=False
        while not succ:
            try:
                start = time.time()
                response = gemini_model.generate_content(
                    messages,
                    generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    response_schema=create_planner_response(
                        frontier_node_list, 
                        room_node_list, 
                        region_node_list, 
                        object_node_list, 
                        Answer_options)
                        )
                )

                print(f"Time taken for planning next step: {time.time()-start}s")
                if (True): # If the model refuses to respond, you will get a refusal message
                    succ=True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 45s")
                time.sleep(45)
        
        
        json_response = response.text
        response_dict = json.loads(json_response)
        step_out = response_dict["steps"][0]
        sg_desc = response_dict["scene_graph_description"]
        if self._use_image:
            img_desc = response_dict["image_description"]
        else:
            img_desc = ' '

        question_type = response_dict["question_type"]
        answer = response_dict['answer']

        if step_out:
            step = {}
            step_type = list(step_out.keys())[0]
            step['step_type'] = step_type
            if (step_type == 'Goto_object_node_step'):
                step['choice'] = step_out[step_type]['object_id']
                step['value']= step_out[step_type]['object_name']
                step['explanation'] = step_out[step_type]['explanation_obj']
                step['room'] = step_out[step_type]['room_name']
                step['explanation_room'] = step_out[step_type]['explanation_room']
            elif (step_type == 'Goto_frontier_node_step'):
                step['choice'] = step_out[step_type]['frontier_id']
                step['explanation'] = step_out[step_type]['explanation_frontier']
            # else:
            #     step['choice'] = step_out[step_type]['answer']
            #     step['value'] = step_out[step_type]['value']
            #     step['explanation'] = step_out[step_type]['explanation_ans']
            #     step['is_confident'] = step_out[step_type]['is_confident']
            #     step['confidence_level'] = step_out[step_type]['confidence_level']
        else:
            step = None
         
        return step, img_desc, sg_desc, question_type, answer
    

    def get_next_action(self):        
        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_state_prompt = self.get_current_state_prompt(self.sg_sim.scene_graph_str, agent_state)

        sg_desc=''
        step, img_desc, sg_desc, question_type, answer = self.get_gemini_output(current_state_prompt)

        print(f'At t={self._t}: \n Step: {step}, \n Answer: {answer}, \n Question type: {question_type}')
        # Saving outputs to file
        self._outputs_to_save.append(f'At t={self._t}: \n \
                                        Agent state: {agent_state} \n \
                                        VLM step: {step} \n \
                                        Image desc: {img_desc} \n \
                                        Scene graph desc: {sg_desc} \n \
                                        Question type: {question_type} \n')
        self.full_plan = ' '.join(self._outputs_to_save)
        with open(self._output_path / "llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        if step is None or step['choice'] == 'Do not choose this option. No more frontiers left.':
            return None, None, answer['is_confident'], answer['confidence_level'], answer['answer']
        

        if self._add_history:
            self.update_history(agent_state, step, question_type)

        self._t += 1

        target_pose = self.sg_sim.get_position_from_id(step['choice'])
        target_id = step['choice']
        return target_pose, target_id, answer['is_confident'], answer['confidence_level'], answer['answer']
