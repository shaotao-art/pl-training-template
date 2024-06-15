from transformers import AutoModelForSequenceClassification

def get_model(model_config):
    model = AutoModelForSequenceClassification.from_pretrained(model_config.model_name, 
                                                            num_labels=model_config.num_classes)
    return model