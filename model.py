from transformers import DebertaForSequenceClassification

def get_model(model_config):
    model = DebertaForSequenceClassification.from_pretrained(model_config.model_name, 
                                                            num_labels=model_config.num_classes)
    return model