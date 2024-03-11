from flask import Flask, request
import json
from typing import Optional
app = Flask(__name__)
from dataclasses import dataclass
from chatwrap import ChatWrapExl2, codegen_prompt
import torch
import gc
import traceback
import sys
@dataclass
class RequestInit:
    code: str
    line: int 
    col: int
    n_tokens: int=32
    language: str = ""
    custom_prompt: str = ""

@dataclass
class RequestLoad:
    model_id: str = ""
    
@dataclass
class Globals:
    model: Optional[ChatWrapExl2] = None
    model_id: str = "LoneStriker_zephyr-7b-beta-8.0bpw-h6-exl2"

G = Globals()

def do_load_model():
    assert G.model_id
    G.model = ChatWrapExl2(f"~/models/{G.model_id}")

@app.route('/load', methods=['POST'])
def load_model():
    print("*** Loading the model")
    G.model = None
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    d = request.get_json(force=True)
    # TODO: sanitize model id
    G.model = ChatWrapExl2(f"~/models/{G.model_id}")
    do_load_model()
    return json.dumps({'status':'done'})

@app.route('/gen', methods=['POST'])
def simple_gen():
    if not G.model:
        do_load_model()
    d = request.get_json(force=True)
    init_request = RequestInit(**d)
    if not G.model:
        result = "(No model loaded)"
    else:
        try:
            prompt = codegen_prompt(init_request.code, init_request.line, init_request.col)
            result = G.model.generate(prompt, n_tokens=init_request.n_tokens, return_new_only=True)
            delete_from = result.find("\n```")
            if delete_from != -1:
                result = result[:delete_from]
            result = result[init_request.col:]

        except Exception as e:
            result = "(INTERNAL ERROR)"
            traceback.print_exc()
            print(f"Exception occured during handling of\n{json.dumps(d, indent=2)}", file=sys.stderr)


    return ({"generation":result})


