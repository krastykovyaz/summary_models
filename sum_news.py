from transformers import MBartTokenizer, MBartForConditionalGeneration
import pandas as pd
# import dill

class ModelGusevSum:
    def __init__(self, ):
        self.model_name = "IlyaGusev/mbart_ru_sum_gazeta"
        self.tokenizer = MBartTokenizer.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        
    
    def summary_title(self, title, maxlength):
        if maxlength > 1000:
            return "Let's try small length of sequence !"
        input_ids = self.tokenizer(
        [title],
        max_length=maxlength,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        )["input_ids"]
        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )[0]

        summ = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return summ
        

if __name__=='__main__':
    model = ModelGusevSum()
    t = """В ближайшие часы с сохранением днём воскресенья по Удмуртии ожидаются грозы. Противопожарная служба напоминает о мерах предосторожности при прохождении грозы: лучше оставаться дома, закрыть окна, двери; не стойте рядом с высокими объектами (деревья, здания, столбы); не пользуйтесь электроприборами и телефоном; если вы застигнуты грозой на велосипеде или мотоцикле, прекратите движение и переждите грозу на расстоянии примерно 30 метрах от них."""
    print(model.summary_title(t[:1000], 1000))
    # with open('summary_model_22012024.pkl', 'wb') as f:
    #     dill.dump(model, f)