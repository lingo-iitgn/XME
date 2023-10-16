from __future__ import print_function

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import argparse
from rich.pretty import pprint
import os
import time

import pickle

parser = argparse.ArgumentParser(description='Converts a Mend output file to an Excel file.')
parser.add_argument('--algo', type=str, help='Algorithm', default="mend", choices=['mend', 'efk', 'ft'])
parser.add_argument('--model', type=str, help='The model name', choices=['mbert-uncased', 'bloom-560m', 'xlm-roberta'], required=True)
# parser.add_argument('--folder', type=str, help='The folder name')
parser.add_argument('--google', type=bool, help='Insert in Google', default=False)
parser.add_argument('--title-index', type=int, help='Google Sheet start Index', default=None)
parser.add_argument('--select-lang', type=str, help='Select a language', default=None)
# parser.add_argument('--metric', type=str, help='The metric to be used', default="es", choices=['loc', 'es'])
args = parser.parse_args()

if args.select_lang:
    assert args.select_lang in ["english", "spanish", "french", "hindi", "gujarati", "bengali", "tamil", "malayalam", "kannada", "arabic", "chinese", "mixed", "inverse"], "Invalid language"
    assert args.title_index is not None, "Title index is required"
else:
    args.title_index = 2

#########################################################################################################################################
LANGS = ["english", "spanish", "french", "hindi", "gujarati", "bengali", "tamil", "malayalam", "kannada", "arabic", "chinese"]
LANG_MAP = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "hindi": "hi",
    "gujarati": "gu",
    "bengali": "bn",
    "kannada": "kn",
    "malayalam": "ml",
    "tamil": "ta",
    "arabic": "ar",
    "chinese": "zh"
}
FT_LANGS = LANGS + ["mixed"]
if args.model in ["bloom-560m", "xlm-roberta"]:
    FT_LANGS.append("inverse")

ALGO = args.algo
MODEL_NAME = args.model
METRICS = ["es", "loc"]
METRIC_NAME_MAP = {
    "es": "Edit Success",
    "loc": "Locality"
}
# LOGS_DIR = "/home/anonymous-xme/mend/mend/logs-locality"
LOGS_DIR = "/home/anonymous-xme/mend/mend/logs-locality"

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.

SPREADSHEET_ID_MAP = {
    "mend": "1gsMiV3oJGe9V9936vMQ_qDYB0GKH2BjW0-hYFaZTiug",
    "efk": "1OTJREbhpYEcxLYeTanqk9Xxe-LIcesf1tz0H_WyDYm8",
    "ft": "1zXRJi41Vw3CZ_6hdgs4LiEjYVpo7KAyDNu9hapev7lo"
}

SPREADSHEET_ID = SPREADSHEET_ID_MAP[ALGO]
SHEET_NAME = f"Ext-{ALGO.upper()}-{MODEL_NAME.split('-')[0].upper()}"
RANGE_NAME = f"{SHEET_NAME}!"
TITLE_INDEX = args.title_index
##########################################################################################################################################

def set_names(ALGO, MODEL_NAME, lang):
    run_names = [
        # f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_full",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_init_layers_1",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_middle",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_last_layer",
        f"{ALGO}_loc_{MODEL_NAME}_finetuned_{lang}_random"
    ]

    return run_names

FOLDER_NAMES = []
# if not args.folder:
for lang in FT_LANGS:
    FOLDER_NAMES.extend(set_names(ALGO, MODEL_NAME, lang))
# else:
#     FOLDER_NAMES.append(args.folder)
# pprint(FOLDER_NAMES)


def google_sheet(sheet_data):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.

    sheet_data = {
        # "english": { # finetuning language
        #     "bloom-560m-english-init-layers-1": {
                                                # loc: [] # 12 * 12 matrix; first column and row is langs
                                                # es: [] # 12 * 12 matrix; first column and row is langs
                                                # } 
        #     "bloom-560m-english-middle": [] # 12 * 12 matrix; first column and row is langs
        #     ...
        # }
    }

    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()

        # add worksheet if not exists
        

        title_index = TITLE_INDEX
        set_index = title_index+1
        # metric_index = 3
        metric_col_index = {
            "es": ("C", "Q"),
            "loc": ("S", "AG")
        }

        avg_title_index = {
            "es": ("O", "Q"),
            "loc": ("AE", "AG")
        }

        total_avg_col_index = {
            "es": ("D", "N"),
            "loc": ("T", "AD")
        }

        row_avg_col_index = {
            "es": ("D", "N"),
            "loc": ("T", "AD")
        }

        grouped_avg_col_index = { # Column to be averaged for each group
            "es": "O",
            "loc": "AE"
        }

        group_matrix_row_index = [
            [1, 2, 3], # Group 1: English, Spanish, French
            [4, 5, 6], # Group 2: Hindi, Gujarati, Bengali
            [7, 8, 9], # Group 3: Tamil, Malayalam, Kannada
            [10], # Group 4: Arabic
            [11] # Group 5: Chinese
        ]

        if title_index == 2:
            result = sheet.values().update(
                    spreadsheetId=SPREADSHEET_ID, 
                    range=RANGE_NAME+f"A{title_index-1}",
                    valueInputOption="USER_ENTERED",
                    body={"values": [["Fine-tuning Language"]]},
                    
            ).execute()

            for k in METRICS:
                result = sheet.values().update(
                        spreadsheetId=SPREADSHEET_ID, 
                        range=RANGE_NAME+f"{avg_title_index[k][0]}{title_index-1}:{avg_title_index[k][1]}{title_index-1}",
                        valueInputOption="USER_ENTERED",
                        body={"values": [["Row Average", "Group Average", "Matrix Average"]]},
                        
                ).execute()

        for idx, (ft_lang, sets) in enumerate(sheet_data.items()):
            if args.select_lang and ft_lang != args.select_lang:
                continue
            # Finetuning Language
            result = sheet.values().update(
                spreadsheetId=SPREADSHEET_ID, 
                range=RANGE_NAME+f"A{title_index}",
                valueInputOption="USER_ENTERED",
                body={"values": [[ft_lang.capitalize()]]},
                
            ).execute()

            for set_name, eval_matrix in sets.items():
                # Set Name
                result = sheet.values().update(
                    spreadsheetId=SPREADSHEET_ID, 
                    range=RANGE_NAME+f"B{set_index}",
                    valueInputOption="USER_ENTERED",
                    body={"values": [[set_name]]}
                ).execute()

                for metric, matrix in eval_matrix.items():
                    # Metric
                    result = sheet.values().update(
                        spreadsheetId=SPREADSHEET_ID, 
                        range=RANGE_NAME+f"{metric_col_index[metric][0]}{set_index}",
                        valueInputOption="USER_ENTERED",
                        body={"values": [[METRIC_NAME_MAP[metric]]]}
                    ).execute()

                    # Matrix
                    # Append Row Average Formula 
                    for i in range(1, len(matrix)):
                        matrix[i].append(f"=AVERAGE({row_avg_col_index[metric][0]}{set_index+i+1}:{row_avg_col_index[metric][1]}{set_index+i+1})")

                    # Append Group Average Formula
                    for i in range(len(group_matrix_row_index)):
                        formula = f"=AVERAGE("
                        j = None
                        for j in group_matrix_row_index[i]:
                            formula += f"{grouped_avg_col_index[metric]}{set_index+j+1},"
                        formula = formula[:-1] + ")"
                        matrix[j].append(formula)

                    # Append Matrix Average Formula
                    matrix[-1].append(
                        f"=AVERAGE({total_avg_col_index[metric][0]}{set_index+1}:{total_avg_col_index[metric][1]}{set_index+len(matrix)-1})"
                    )

                    # print(f"Matrix for {ft_lang} - {set_name} - {metric}")
                    # pprint(matrix)

                    result = sheet.values().update(
                        spreadsheetId=SPREADSHEET_ID, 
                        range=RANGE_NAME+f"{metric_col_index[metric][0]}{set_index+1}:{metric_col_index[metric][1]}{set_index+len(matrix)}",
                        valueInputOption="USER_ENTERED",
                        body={"values": matrix}
                    ).execute()

                # metric_index += len(matrix) + 2
                set_index += len(matrix) + 2

            title_index += 58
            set_index += 2

            print(f"Done with {ft_lang}!")
            if idx != 0 and idx % 2 == 1 and not args.select_lang:
                print("Sleeping for 60 seconds!")
                time.sleep(60*2)   
            print("Continuing...")

    except HttpError as err:
        print(err)


def get_row_data(file_name, ed_lang):
    """
        metric: "loc" means locality or "es" means edit success
    """
    
    with open(file_name, "r") as f:
        lines = f.readlines()

    data = {}
    for metric in METRICS:
        row_data = [ed_lang.capitalize()]
        for lang in LANGS: # Evaluation Language
            for line in lines:
                if line.startswith(f"{metric}_{LANG_MAP[lang]}"):
                    row_data.append(float(line.split(":")[1].strip()))
        data[metric] = row_data

    return data


if __name__ == '__main__':
    
    sheet_data = {
        # "english": { # finetuning language
        #     "bloom-560m-english-init-layers-1": {
                                                # loc: [] # 12 * 12 matrix; first column and row is langs
                                                # es: [] # 12 * 12 matrix; first column and row is langs
                                                # } 
        #     "bloom-560m-english-middle": [] # 12 * 12 matrix; first column and row is langs
        #     ...
        # }
    }

    for folder_name in FOLDER_NAMES:

        # FT Language
        temp = folder_name.split("_")
        ft_lang = temp[4]
        set_name = "-".join(temp[5:])
        # print(f"Set name: {set_name}")
        # print(f"FT Language: {ft_lang}")
        # print(f"Folder name: {folder_name}")

        matrix = {
            k: [] for k in METRICS
        }

        # Add first row as languages
        for k, v in matrix.items():
            v.append(["E/V"]+[l.capitalize() for l in LANGS])

        for lang in LANGS: # Edit Language
            path = f"{LOGS_DIR}/{folder_name}/{lang}.txt"
            if os.path.exists(path):
                row_data = get_row_data(path, lang)
                # pprint(row_data)
            else: # No file exists
                row_data = {
                    k: [] for k in METRICS
                }
                for k, v in row_data.items():
                    v.extend([lang.capitalize()] + [-1 for _ in range(len(LANGS))])
                
        
            for k, v in row_data.items():
                matrix[k].append(v)

            

        # pprint(matrix)

        set_dict = sheet_data.get(ft_lang, {})
        set_dict[set_name] = matrix
        sheet_data[ft_lang] = set_dict

    pprint(sheet_data)

    pickle.dump(sheet_data, open(f"/home/anonymous-xme/mend/mend/es_loc_out_plks/{ALGO}-{MODEL_NAME}.pkl", "wb"))

    if args.google:
        google_sheet(sheet_data)