#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tornado.ioloop
import tornado.web
import json
# from RemoteAgent import *
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

histories = {}
# models = {}
usrStacks = {}


tokenizer = AutoTokenizer.from_pretrained("../../models/log/dialogpt-m-ft-alexa-our_attn-full-v100_t3")
model = AutoModelWithLMHead.from_pretrained("../../models/log/dialogpt-m-ft-alexa-our_attn-full-v100_t3")

# eos_token_id = tokenizer.encode("_eos")[0]
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
# eos_token_id = tokenizer.eos_token_id

print("loading is done")

# class API(object):
#     def __init__(self):
#         # self.history = []
#         self.model = DialoGPTBaseline("fine_tuned")
    
#     def GetResponse(self,text):
#         # slu = {"act":"dialog_act","slot":"named_entity"} 
#         sysUtter = self.model.get_response(text)
#         # imageurl = "https://skylar.speech.cs.cmu.edu/image/movie.jpg"
#         return sysUtter
# In[25]:


class DialoGPTBaseline():
    def __init__(self):
        # self._load_model(model_opt)
        self.chat_history_ids = None
        self.step = 0
    
    # def _load_model(self, model_opt):
    #     if model_opt == 'fine_tuned':
    #         self.tokenizer = AutoTokenizer.from_pretrained("./output-dialogpt-medium2/1")
    #         self.model = AutoModelWithLMHead.from_pretrained("./output-dialogpt-medium2/1")
    #         self.eos_token_id = self.tokenizer.encode(self.tokenizer.additional_special_tokens[3])[0]
    #     elif model_opt == 'pretrained':
    #         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    #         self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
    #         self.eos_token_id = self.tokenizer.eos_token_id
    #     return 
    
    def get_response(self, text):
        print("get_response")
        new_user_input_ids = tokenizer.encode( text + tokenizer.eos_token, return_tensors='pt')
        # print("1")
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.step > 0 else new_user_input_ids
        # print("2")
        # print(bot_input_ids.shape)
        self.chat_history_ids = model.generate(bot_input_ids, 
                                                    max_length=1024, do_sample=True, top_k=20, top_p=0.95,
                                                    pad_token_id=tokenizer.pad_token_id, use_cache=False)
        # print("3")
        response = tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # if self.chat_history_ids.shape[-1] > 70:
        #     self.chat_history_ids = self.chat_history_ids[:, -30:]
        self.step += 1
        print(response)
        print("out")
        return response.strip()

def reply(history):
    return ' '.join(history[-1].split()[::-1])
#     return model

class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def options(self):
        pass

    def post(self):
        body = json.loads(self.request.body.decode())
        if body["text"] == "SSTTAARRTT":
            histories[body["userID"]] = []
            usrStacks[body["userID"]] = DialoGPTBaseline()
        else:
            histories[body["userID"]].append(body["text"])
#             response = reply(histories[body["userID"]])

            response = usrStacks[body["userID"]].get_response(body["text"])
    
            histories[body["userID"]].append(response)
            self.write(json.dumps({"body": json.dumps({"utterance": response})}))

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8891)
    tornado.ioloop.IOLoop.current().start()

# %%
