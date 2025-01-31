from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmpathyScoreDeployedCL:
    """
    Uses a pretrained empathy scoring model (e.g., bert_empathy) to evaluate empathy scores for generated text.
    """
    def __init__(self, same_length=False, score_change=False):
        self.same_length = same_length
        model_path = "models/bert-empathy"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.score_change = score_change
        if self.score_change:
            self.score = self.score_relative
        else:
            self.score = self.score_absolute
    def preprocess_batch(self, sources, decoded):
        c_prompts = []
        u_responses = []
        for source, decod in zip(sources, decoded):
            sp = source.split("[SEP]")
            if len(sp) != 2:
                print("Error, formatting must be wrong")
                print("source:", source)
                continue
            client_prompt = sp[0].strip()
            c_prompts += [ client_prompt  ]
            u_responses += [ sp[1] ]
        max_output_length = 160 
        c_prompts_u_responses = self.tokenizer(c_prompts, u_responses, padding='longest',truncation=True,return_tensors='pt')
        c_prompts_model_responses = self.tokenizer(c_prompts, decoded, padding='longest',truncation=True,return_tensors='pt')
        c_prompts_u_responses = c_prompts_u_responses.to(self.device)
        c_prompts_model_responses = c_prompts_model_responses.to(self.device)
        return c_prompts_u_responses, c_prompts_model_responses
    def sim_func(self, a,b):
        return (self.cos_sim(a,b)+1.0)/2.0
    def score_relative(self, sources, generateds, partial=False, printing=False, **kwargs):
        up_to_length = None
        if self.same_length or partial:
            up_to_length = len(self.tokenizer.encode(generateds[0]))
        c_prompts_u_responses, c_prompts_model_responses = self.preprocess_batch(sources, generateds)
        with torch.no_grad():
            score_pm = self.model(**c_prompts_model_responses)[0].detach().squeeze()
            score_pu = self.model(**c_prompts_u_responses)[0].detach().squeeze()
        scores = score_pm - score_pu 
        scores = scores.tolist()
        if printing:
            print("[empathy_change]", scores)
        return {"scores": scores  } 
    def score_absolute(self, sources, generateds, partial=False, printing=False, **kwargs):
        up_to_length = None
        if self.same_length or partial:
            up_to_length = len(self.tokenizer.encode(generateds[0]))
        c_prompts_u_responses, c_prompts_model_responses = self.preprocess_batch(sources, generateds)
        with torch.no_grad():
            score_pm = self.model(**c_prompts_model_responses)[0].detach()
        scores = torch.clamp(score_pm, 0, 1).squeeze()
        scores = scores.tolist()
        if printing:
            print("[empathy]", scores)
        return {"scores": scores }
