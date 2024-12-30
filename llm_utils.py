from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json



def process(message, controller, st):
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.1
    )
    prompt = PromptTemplate(
        input_variables=["command"],
        template = ''' 
            You are a AI chatbot which is a TODO list bot, user will give you the query and you have to intelligently classify the query and also the task which they add or complete.
            You will be given a command in a form of query command can be of three types i.e - ADD, COMPLETE, SHOW
            The command will be in normal english language, you have to decide with your super intelligent power what command it is.
            With these command there can be a task associated to it. like Add green juice to the list so here green juice is the task.
            Examples
            1.Query:- Add morning walk to list => action: ADD, task: morning walk
            2.Query:- Complete morning walk => action: COMPLETE, task: morning walk
            3.Query:- homework is done => action: COMPLETE, task: homework
            4.Query:- display due tasks => action: SHOW, task: ""
            5.Query:- list all tasks => SHOW, task: ""
            You have to give the output strictly in json format with NO PREAMBLE with two keys - "action", "task"

            Here is the query - {query}
        '''
    )

    chain = LLMChain(
        llm = llm,
        prompt = prompt
    )
    query = message
    response = chain.run(query)
    resp_json = json.loads(response)

    action = resp_json.get("action")
    task = resp_json.get("task")
    tasks = controller.get('tasks')
    if tasks is None:
        tasks = []
    print("these are the tasks", tasks)
    tasks, answer = process_action(action, task, tasks, llm)
    controller.set('tasks', tasks)

    st.text(answer)



def process_action(action, task, tasks, llm):
    answer = ""
    if action == "ADD":
        tasks.append({"task": task, "status": "Pending"})
        answer = f"{task} added to the list"
    elif action == "COMPLETE":
        all_tasks = []
        for t in tasks:
            all_tasks.append(t["task"])

        prompt_complete = PromptTemplate(
            input_variables=['task_list', 'task'],
            template= '''
            You are the super intelligent AI, so I will give the tasks list and also one task 
            You have to find the most correct task from that list which relates with task, if you find nothing matches then give a empty string
            task_list = {task_list}
            task = {task}
            Just give me the most correct task from task list with just task name as it is with NO PREAMBLE.
            '''
        )
        complete_chain = LLMChain(llm=llm,prompt=prompt_complete)
        predicted_task = complete_chain.run({
            "task_list": all_tasks,
            "task": task
        })
        for t in tasks:
            if t["task"] == predicted_task:
                t["status"] = "Completed"
                answer = f"{predicted_task} marked as completed."
                break
    elif action == "SHOW":
        if len(tasks) == 0:
            answer = "No tasks in the list."
        else:
            answer = "Here are your all tasks.\n"
            for t in tasks:
                answer += f'''{t["task"]} - {t["status"].upper()}'''
    return tasks, answer
