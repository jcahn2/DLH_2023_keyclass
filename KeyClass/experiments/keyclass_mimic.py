
# ---------------------------------------------------------------------------- #
#                            KEYCLASS PYTHON SCRIPT                            #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import sys
sys.path.append('../keyclass/')
sys.path.append('../scripts/')

import argparse
import label_data, encode_datasets, train_downstream_model
import torch
import pickle
import numpy as np
import os
from os.path import join, exists
from datetime import datetime
import utils
import models
import create_lfs
import train_classifier
from transformers import AutoTokenizer, AutoModel
import gc
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.utils import shuffle
import itertools
from sklearn.metrics import precision_recall_curve
import pandas as pd

# configurations
random_seed = 0


# ---------------------------------------------------------------------------- #
#                                   gpu clean                                  #
# ---------------------------------------------------------------------------- #
def clean_gpu(model=None):
    gc.collect()
    torch.cuda.empty_cache()
    if model:
        del model

# ---------------------------------------------------------------------------- #
#                              Data Loading, Preprocessing                     #
# ---------------------------------------------------------------------------- #      
def load_data(train_path = '../fastag_data/icd9NotesDataTable_train.csv', 
              test_path = '../fastag_data/icd9NotesDataTable_valid.csv', 
              out_path = '../data/mimic/', 
              max_length = 1000, 
              random_state = 1234, 
              sample_size = 0.1):
    train_path = train_path
    test_path = test_path
    codeIdx = 9
    textIdx = 6
    utils.load_and_process_data(train_path, 
                                test_path, 
                                out_path, 
                                max_length=max_length, 
                                random_state=random_state, 
                                sample_size=sample_size)
# ---------------------------------------------------------------------------- #
#                              EMBEDDING FUNCTION                              #
# ---------------------------------------------------------------------------- #
def create_embeddings(args):
    # choose between custom encoder and regular Encoder in args config file
    if args['use_custom_encoder']:
        model = models.CustomEncoder(pretrained_model_name_or_path=args['base_encoder'], 
            device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        model = models.Encoder(model_name=args['base_encoder'], 
            device='cuda' if torch.cuda.is_available() else 'cpu')

    # split data
    for split in ['train', 'test']:
        sentences = utils.fetch_data(dataset=args['dataset'], split=split, path=args['data_path'])
        embeddings = model.encode(sentences=sentences, batch_size=args['end_model_batch_size'], 
                                    show_progress_bar=args['show_progress_bar'], 
                                    normalize_embeddings=args['normalize_embeddings'])
        # save embedding results
        with open(join(args['data_path'], args['dataset'], f'{split}_embeddings.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)
    
    return model



# ---------------------------------------------------------------------------- #
#                      PREPARE DATA FOR LABELING FUNCTION                      #
# ---------------------------------------------------------------------------- #
def split_data_lf(args, train_labels_filename):
    train_text = utils.fetch_data(dataset=args['dataset'], path=args['data_path'], split='train')

    training_labels_present = False
    if exists(join(args['data_path'], args['dataset'], train_labels_filename)):
        with open(join(args['data_path'], args['dataset'], train_labels_filename), 'r') as f:
            y_train = f.readlines()
        y_train = np.array([[int(i) for i in sub.strip().split()] for sub in y_train], dtype=object)
        
        training_labels_present = True
    else:
        y_train = None
        training_labels_present = False
        print('No training labels found!')

    with open(join(args['data_path'], args['dataset'], f'train_embeddings.pkl'), 'rb') as f:
        X_train = pickle.load(f)

    # Convert to MultiLabel format
    mlb = MultiLabelBinarizer()
    y_train_ml = mlb.fit_transform(y_train)

    # Print dataset statistics
    print(f"Getting labels for the {args['dataset']} data...")
    print(f'Size of the data: {len(train_text)}')
    if training_labels_present:
        print('Class distribution', np.unique(np.hstack(y_train.flatten()), return_counts=True))
    class_balance = np.unique(np.hstack(y_train.flatten()), return_counts=True)[1] / np.unique(np.hstack(y_train.flatten()), return_counts=True)[1].sum()
    # Load label names/descriptions
    label_names = []
    for a in args:
        if 'target' in a: label_names.append(args[a])
        
    return train_text, label_names, y_train, y_train_ml, class_balance, training_labels_present

# ---------------------------------------------------------------------------- #
#                            TRAIN LABELING FUNCTION                           #
# ---------------------------------------------------------------------------- #
def train_lf(args, train_labels_filename):
    
    train_text, label_names, y_train, y_train_ml, class_balance, training_labels_present = split_data_lf(args, train_labels_filename)
    # Creating labeling functions
    labeler = create_lfs.CreateLabellingFunctions(custom_encoder=args['use_custom_encoder'],
                                                base_encoder=args['base_encoder'], 
                                                device=torch.device(args['device']),
                                                label_model=args['label_model'])
    # obtain predicted probabilities and labels from label model
    proba_preds,y_preds = labeler.get_labels(text_corpus=train_text, label_names=label_names, max_df = 1.0, min_df=0.001, 
                                    ngram_range=(1,1), topk=args['topk'], y_train=y_train_ml, 
                                    label_model_lr=args['label_model_lr'], label_model_n_epochs=args['label_model_n_epochs'], 
                                    verbose=True, n_classes=args['n_classes'], class_balance=class_balance, min_topk=False)

    y_train_pred = y_preds


    # Save the predictions
    if not os.path.exists(args['preds_path']): os.makedirs(args['preds_path'])
    with open(join(args['preds_path'], f"{args['label_model']}_proba_preds.pkl"), 'wb') as f:
        pickle.dump(proba_preds, f)

    # Print statistics
    print('Label Model Predictions: Unique value and counts', np.unique(
        y_preds.flatten(), return_counts=True
    ))
    if training_labels_present:
        print('Label Model Training Accuracy', np.mean([(y_train_pred[i] in labels) for i,labels in enumerate(y_train)]))


        # Log the metrics
        training_metrics_with_gt = utils.compute_metrics(y_preds=y_train_pred, y_true=y_train, average=args['average'])
        utils.log(metrics=training_metrics_with_gt, filename='label_model_with_ground_truth', 
            results_dir=args['results_path'], split='train')

    return labeler, class_balance

# ---------------------------------------------------------------------------- #
#                          TRAIN DOWNSTREAM CLASSIFIER                         #
# ---------------------------------------------------------------------------- #
def train_model(
            args, 
            classifier, 
            X_train_embed_masked,
            y_train_lm_masked,
            X_test_embed,
            sample_weights_masked, 
            batch_size, 
            raw_text=False, 
            weight_decay=0, 
            lr=0.001):
    
    print('\n===== Training the TCN downstream classifier =====\n')
    X_train_embed_masked, y_train_lm_masked = shuffle(X_train_embed_masked, y_train_lm_masked, random_state=2)

    model = train_classifier.train_multi_label(model=classifier, 
                                device=torch.device(args['device']),
                                X_train=X_train_embed_masked, 
                                y_train=y_train_lm_masked,
                                sample_weights=sample_weights_masked if args['use_noise_aware_loss'] else None, 
                                epochs=args['end_model_epochs'], 
                                batch_size=batch_size, 
                                criterion=eval(args['criterion']), 
                                raw_text=False, 
                                lr=lr, 
                                weight_decay=weight_decay,
                                patience=args['end_model_patience'])

    end_model_preds_train = model.predict_proba(torch.from_numpy(X_train_embed_masked), batch_size=batch_size, raw_text=raw_text)
    end_model_preds_test = model.predict_proba(torch.from_numpy(X_test_embed), batch_size=batch_size, raw_text=raw_text)
    
    return model, end_model_preds_train, end_model_preds_test

# --------------------- CREATE FEEDFORWARDFLEXIBLE MODEL --------------------- #
def create_FeedForward(args, encoder, h_sizes):
    classifier = models.FeedForwardFlexible(
        encoder_model=encoder,
        h_sizes=h_sizes, 
        activation=eval(args['activation']),
        device=torch.device(args['device'])
    )
    
    return classifier

# ----------------------------- CREATE TCN MODEL ----------------------------- #
def create_TCN(args, 
               encoder, 
               num_inputs,
               channels1, 
               channels2, 
               h_sizes, 
               kernel_size,
               dropout,  
               batch_size):
    classifier = models.FeedForwardTCN(
        encoder_model=encoder,
        num_inputs=num_inputs, 
        num_channels1=[channels1, channels1, channels1], 
        num_channels2=[channels2, channels2, channels2],
        h_sizes=h_sizes,
        kernel_size=kernel_size, 
        dropout=dropout, 
        batch_size=batch_size, 
        device=torch.device(args['device']))
    
    return classifier

# ---------------------------------------------------------------------------- #
#                     MAKE AND TRAIN DOWNSTREAM CLASSIFIER                     #
# ---------------------------------------------------------------------------- #
def downstream_classifier(args, class_balance, classifier_type='FeedForwardFlexible'):
    
    # Set random seeds
    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    batch_size = args['batch_size']
    lr = args['classifier_model_lr']

    X_train_embed_masked, y_train_lm_masked, y_train_masked, \
        X_test_embed, y_test, training_labels_present, \
        sample_weights_masked, proba_preds_masked = train_downstream_model.load_data_all(args, class_balance=class_balance, max_num=2000)


    if args['use_custom_encoder']:
        encoder = models.CustomEncoder(pretrained_model_name_or_path=args['base_encoder'], device=args['device'])
    else:
        encoder = models.Encoder(model_name=args['base_encoder'], device=args['device'])
    
    # create classifier
    if classifier_type == 'FeedForwardFlexible':
        classifier = create_FeedForward(args=args, encoder=encoder, h_sizes=args['h_sizes_FF'])
    elif classifier_type == 'TCN':
        classifier = create_TCN(
            args=args, 
            encoder=encoder, 
            num_inputs=args['num_TCN_inputs'], 
            channels1=args['num_TCN_channels'][0], 
            channels2=args['num_TCN_channels'][1], 
            h_sizes=args['h_sizes_TCN'], 
            kernel_size=args['kernel_size'], 
            dropout=args['dropout'], 
            batch_size=batch_size
        )

    # train_classifier
    model, end_model_preds_train, end_model_preds_test = train_model(
        args=args, 
        classifier=classifier,
        X_train_embed_masked=X_train_embed_masked,
        y_train_lm_masked=y_train_lm_masked,
        X_test_embed=X_test_embed, 
        sample_weights_masked=sample_weights_masked, 
        batch_size=batch_size, 
        raw_text=False, 
        weight_decay=0, 
        lr=lr
    )
    
    return model, y_test, end_model_preds_train, end_model_preds_test
    

# ---------------------------------------------------------------------------- #
#                       LOAD EMBEDDED DATA FOR SELF-TRAIN                      #
# ---------------------------------------------------------------------------- #
def load_embed(args):
    with open(
            join(args['data_path'], args['dataset'], f'train_embeddings.pkl'),
            'rb') as f:
        X_train_embed = pickle.load(f)
    with open(join(args['data_path'], args['dataset'], f'test_embeddings.pkl'),
            'rb') as f:
        X_test_embed = pickle.load(f)
    
    return X_train_embed, X_test_embed

# ---------------------------------------------------------------------------- #
#                                  SELF-TRAIN                                  #
# ---------------------------------------------------------------------------- #
def self_train(args, model, y_test):
    X_train_embed, X_test_embed = load_embed(args)

    model = train_classifier.self_train(model=model, 
                                        X_train=X_train_embed, 
                                        X_val=X_test_embed, 
                                        y_val=y_test, 
                                        device=torch.device(args['device']), 
                                        lr=eval(args['self_train_lr']), 
                                        weight_decay=eval(args['self_train_weight_decay']),
                                        patience=args['self_train_patience'], 
                                        batch_size=args['self_train_batch_size'], 
                                        q_update_interval=args['q_update_interval'],
                                        self_train_thresh=eval(args['self_train_thresh']), 
                                        print_eval=True,
                                        raw_text=False, 
                                        train_multilabel=True)
    
    X_test_embed = torch.from_numpy(X_test_embed)
    
    model.eval()
    test_result = model.forward(torch.tensor(X_test_embed).to("cuda"), raw_text=False).cpu().detach().numpy()
    train_result = model.forward(torch.tensor(X_train_embed).to("cuda"), raw_text=False).cpu().detach().numpy()
    return model, test_result, train_result

# ---------------------------------------------------------------------------- #
#                     Predict on Best Threshold                                #
# ---------------------------------------------------------------------------- #

def predict_on_cust_thresholds(model, y_pred_train, y_pred_test, y_train, y_test):
   

    
    repeats = [len(x) for x in y_train]
    y_train_multi = np.array(list(itertools.chain.from_iterable(y_train)))
    y_pred_train_multi = np.repeat(y_pred_train,repeats,axis=0)
    
    _, n_classes = y_pred_train_multi.shape
    overall_thresholds = []
    for i in range(n_classes):
        
        # Computing best threshold for i-th class
        precision, recall, thresholds = precision_recall_curve(y_train_multi, y_pred_train_multi[:, i], pos_label=i)

        # compute f-1
        f1 = 2 * precision * recall / (precision + recall)

        # pick up the best threshold's index
        best_idx = np.argmax(f1)
        overall_thresholds.append(thresholds[best_idx])
    
    overall_thresholds = np.array(overall_thresholds)
    y_pred_bool = y_pred_test > overall_thresholds[None,:]
    return y_pred_bool

# ---------------------------------------------------------------------------- #
#                                   Score                                      #
# ---------------------------------------------------------------------------- #

def precision_recall_accuracy(y_pred, y_true):
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(y_true)
    actual_pos_sum = y_true.sum(axis=0)
    pred_pos_sum = y_pred.sum(axis=0)
    pred_actual_intersect = (y_true&y_pred).sum(axis=0)
    
    class_precision = np.nan_to_num(pred_actual_intersect/pred_pos_sum)
    overall_precision = pred_actual_intersect.sum() / pred_pos_sum.sum()
    class_recall = np.nan_to_num(pred_actual_intersect/actual_pos_sum)
    overall_recall = pred_actual_intersect.sum() / actual_pos_sum.sum()
    
    class_accuracy = (y_true==y_pred).sum(axis=0) / y_true.shape[0]
    overall_accuracy = (y_true==y_pred).sum() / (y_true.shape[0] * y_true.shape[1])
    
    return class_precision, class_recall, class_accuracy, overall_precision, overall_recall, overall_accuracy

# ---------------------------------------------------------------------------- #
#                                   Write                                      #
# ---------------------------------------------------------------------------- #


def write_to_csv(tcn_scores, ff_scores, outpath):
    categories = [
        'Infectious', 
        'Neoplasia',
        'Endo-Immune',
        'Blood',
        'Mental',
        'Nervous',
        'Sense organs',
        'Circulatory',
        'Respiratory',
        'Digestive',
        'Genitourinary',
        'Pregnancy',
        'Skin',
        'Musculoskeletal',
        'Congenital',
        'Perinatal',
        'Injury'
    ]
    tcn_class = pd.DataFrame(index=categories)
    ff_class = pd.DataFrame(index=categories)
    
    overall = pd.DataFrame(index=['Precision', 'Recall', 'Accuracy'])
    tcn_class['Precision'] = tcn_scores[0]
    tcn_class['Recall'] = tcn_scores[1]
    tcn_class['Accuracy'] = tcn_scores[2]
    overall['TCN Classifier'] = tcn_scores[3:]
    
    
    ff_class['Precision'] = ff_scores[0]
    ff_class['Recall'] = ff_scores[1]
    ff_class['Accuracy'] = ff_scores[2]
    overall['FF Classifier'] = ff_scores[3:]
    
    ff_class.to_csv(f'{outpath}_ff.csv', index=False)
    tcn_class.to_csv(f'{outpath}_tcn.csv', index=False)
    overall.to_csv(f'{outpath}_overall.csv', index=False)


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def main():
    load_data(sample_size = 1.0)
    configs = [r'../config_files/config_mimic_bert.yml', r'../config_files/config_mimic_mpnet.yml']
    outputs = ['../results/bert','../results/mpnet']
    for i, config in enumerate(configs):
        config_file_path = config
        clean_gpu()
        # ----------------------------- parse config file ---------------------------- #
        args = utils.Parser(config_file_path=config_file_path).parse()


        # ----------------- pass through encoder to create embeddings ---------------- #
        embedding_model = create_embeddings(args)
        # clean gpu cache and del model
        clean_gpu(embedding_model)

        # --------------- use embeddings to create labelling functions --------------- #
        # ----- that generate probabilstic labels that will be used as inputs to ----- #
        # ---------------------- train the downstream classifier --------------------- #
        train_labels_filename = 'train_labels_all.txt'
        lf_model, class_balance = train_lf(args=args, train_labels_filename=train_labels_filename)

        clean_gpu(lf_model)


        # ----------------- train both FeedFoward and TCN classifiers ---------------- #


        # train FeedForward

        modelFF, y_test_FF, end_modelFF_preds_train, end_modelFF_preds_test = downstream_classifier(
            args=args, 
            class_balance=class_balance, 
            classifier_type='FeedForwardFlexible',
        )

        # train TCN

        modelTCN, y_test_TCN, end_modelTCN_preds_train, end_modelTCN_preds_test = downstream_classifier(
            args=args, 
            class_balance=class_balance, 
            classifier_type='TCN',
        )


        # -------------------------- self-train both models -------------------------- #
        modelFF, test_result_FF, train_result_FF = self_train(
            args=args,
            model=modelFF,
            y_test=y_test_FF
        )
        modelTCN, test_result_TCN, train_result_TCN = self_train(
            args=args,
            model=modelTCN,
            y_test=y_test_TCN
        )

        # ------------------------ produce output and results ------------------------ #

        clean_gpu(modelFF)
        clean_gpu(modelTCN)

        with open(join(args['data_path'], args['dataset'], train_labels_filename), 'r') as f:
            y_train = f.readlines()
        y_train = np.array([[int(i) for i in sub.strip().split()] for sub in y_train], dtype=object)

        test_labels_filename = 'test_labels_all.txt'
        with open(join(args['data_path'], args['dataset'], test_labels_filename), 'r') as f:
            y_test = f.readlines()
        y_test = np.array([[int(i) for i in sub.strip().split()] for sub in y_test], dtype=object)

        tcn_pred = predict_on_cust_thresholds(modelTCN, train_result_TCN, test_result_TCN, y_train, y_test)
        ff_pred = predict_on_cust_thresholds(modelFF, train_result_FF, test_result_FF, y_train, y_test)

        tcn_scores = precision_recall_accuracy(tcn_pred, y_test)
        ff_scores = precision_recall_accuracy(ff_pred, y_test)

        write_to_csv(tcn_scores,ff_scores, outputs[i])

        print('PROCESS COMPLETE')
    
if __name__ == "__main__":
    main()
