import requests
from requests.auth import HTTPBasicAuth
from pprint import pprint

from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


def get_token(api, login, mdp):
    """
    Récupération du token

    :param api: adresse du service
    :param login: str, login à utiliser
    :param mdp: str, mdp à utiliser
    :return: token_type, access_token  (str, str)
    """
    r = requests.post(api + "/token", 
                      data={'username': login, "password": mdp},
                      verify=False)
    token_type = r.json()['token_type']
    access_token = r.json()['access_token']
    return token_type, access_token

def create_session(api, login, mdp):
    """
    Création d'une session portant les headers d'identification

    :param api: adresse du service
    :param login: str, login à utiliser
    :param mdp: str, mdp à utiliser
    :return: request.Session
    """
    token_type, access_token = get_token(api, login, mdp)
    s = requests.Session()
    s.headers.update({'Authorization': f'{token_type} {access_token}'})
    r = s.get(api + "/users/me",verify=False)
    assert r.status_code == 200
    return s

def format_input(input_dict):
    """
    Formattage des inputs pour appel de l'API

    :param input_dict
    :return: json
    """
    j = {
        "cabbi": input_dict.get("cabbi",""),
        "rs_x": input_dict.get("rs_x", ""),
        "actet_x": input_dict.get("actet_x", ""),
        "profs_x": input_dict.get("profs_x", ""),
        "profi_x": input_dict.get("profi_x", ""),
        "profa_x": input_dict.get("profa_x", ""),
        "numvoi_x": input_dict.get("numvoi_x", ""),
        "bister_x": input_dict.get("bister_x", ""),
        "typevoi_x": input_dict.get("typevoi_x", ""),
        "nomvoi_x": input_dict.get("nomvoi_x", ""),
        "cpladr_x": input_dict.get("cpladr_x", ""),
        "clt_x": input_dict.get("clt_x", ""),
        "dlt_x": input_dict.get("dlt_x", ""),
        "plt_x": input_dict.get("plt_x", ""),
        "clt_c_c": input_dict.get("clt_c_c", ""),
        "vardompart_x": input_dict.get("vardompart_x", "")
    }
    
    for key in j:
      if j[key] is None:
        j[key] = ""
       
    return j
    

def call_prediction(api, endpoint, session, json_documents):
    """
    Appel d'un endpoint

    :param api: adresse du service
    :param endpoint: nom du endpoint
    :param session: request.Session initialisée avec create_session
    :param json_documents: [json], documents formattés avec format_input
    :return: dict (réponse du endpoint)
    """
    
    r = session.get(api + "/" + endpoint, 
                    json={"documents": json_documents},
                    verify=False)
    assert r.status_code == 200, f"Error code {r.status_code} on endpoint {api}/{endpoint}: {r.content}"
    return r.json()


def predict_noncodable(api, session, json_documents):
    return call_prediction(api, "noncodable", session, json_documents)

def predict_naf(api, session, json_documents):
    return call_prediction(api, "naf", session, json_documents)

def predict_pcs(api, session, json_documents):
    return call_prediction(api, "pcs", session, json_documents)

def predict_siret(api, session, json_documents):
    return call_prediction(api, "siret", session, json_documents)


if __name__ == "__main__":

    '''
    adresse_api = "http://localhost:8000"
    s = create_session(adresse_api, "admin_recap", "admin_recap")
    '''
    adresse_api = "https://aiee2-prediction-api-test.ouest.innovation.insee.eu" 
    s = create_session(adresse_api, "admin_recap", "admin_recap2")
    
    docs = [
        {"cabbi": "7300014727", "rs_x": "COLLEGE GEORGES MANDEL", "actet_x": "ACCUEIL DES COLLEGIENS EDUCATION", "profs_x": "ASSISTANTE D EDUCATION",
                                      "numvoi_x": "", "typevoi_x": "", "nomvoi_x": "", "clt_x": "SOULAC SUR MER",
                                      "dlt_x": "33", "plt_x": "", "vardompart_x": "", "clt_c_c": "33544"},
        {"cabbi": "7300014728", "rs_x": "ETAT", "actet_x": "PHARES ET BATISES", "profs_x": "TECHNICIEN SUPERIEUR PRINCIPAL",
          "numvoi_x": "0004", "typevoi_x": "QU", "nomvoi_x": "DE CORDOUAN", "clt_x": "",
          "dlt_x": "", "plt_x": "", "vardompart_x": "", "clt_c_c": "33544"}
    ]

    formatted_docs = [format_input(d) for d in docs]
    pprint(predict_noncodable(adresse_api, s, formatted_docs))
    pprint(predict_naf(adresse_api, s, formatted_docs))
    pprint(predict_pcs(adresse_api, s, formatted_docs))
    pprint(predict_siret(adresse_api, s, formatted_docs))
