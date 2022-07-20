"""
2021/01/19

Wrapper pour appeler les api de géocodage

Génère un datagrame avec les infos de géocodage
Génère un fichier csv. dans keep/ pour exporter si besoin les données 

Auteur: Brivaël Sanchez et Yves-Laurent Benichou
"""
import os
import sys
sys.path.append("..")
import csv
import requests
import json
import re
import logging
import pandas as pd
import data_import.bdd as bdd
import elastic as elastic
from training_classes.utils import *
#from functools import lru_cache

#@lru_cache(maxsize = 1024)
def geo(input_file, addok_url=None, path_to_geodata='keep'):
    """
    Fonction de géocodage
    Charge des données à géocoder dans :param input_file:, appelle les api de géocodage (ban, bano, poi)
    et post-processe les résultats

    utilise les fichiers path_to_geodata/geo_cog_old_2018.csv et path_to_geodata/geo_cog_new_2018.csv

    sauvegarde le résultat dans path_to_geodata/geo-outputfile_name au format csv avec les colonnes
    ['cabbi','longitude', 'latitude', 'geo_score', 'geo_type', 'geo_adresse']

    :param input_file: fichier à charger.
                        fichier csv à partir d'une dataframe pandas. Doit contenir :
                        cabbi, numvoie, indrep, typvoie, libvoie, compladr, depcom
    :param addok_url: dict de chemins vers les api de geocodage insee
                        addok_url['ban']  = 'http://api-ban.ouest.innovation.insee.eu/search'
                        addok_url['bano'] = 'http://api-bano.ouest.innovation.insee.eu/search'
                        addok_url['poi']  = 'http://api-poi.ouest.innovation.insee.eu/search'
    :param path_to_geodata: dossier contenant les fichiers data et où le résultat sera sauvé
    """
    # print("begin geo")
    score_min = 0.30

    # URL à appeler pour géocodage BAN et BANO
    if addok_url is None:
        addok_url={}
        addok_url['ban']  = 'http://api-ban.ouest.innovation.insee.eu/search'
        addok_url['bano'] = 'http://api-bano.ouest.innovation.insee.eu/search'
        addok_url['poi']  = 'http://api-poi.ouest.innovation.insee.eu/search'

    # Ouverture de sessions pour requests en keep-alive
    s={}
    GEO_PROXY = os.environ.get('geo_proxy', None)
    for url in addok_url.values():
        s[url]=requests.Session()
        if GEO_PROXY is not None:
            proxies = {
                'http': GEO_PROXY,
                'https': GEO_PROXY,
            }
            s[url].proxies.update(proxies)

    # req. sur l'API de géocodage
    #@lru_cache(maxsize = 1024)
    def geocode(api, params, l4):
        """
        Fonction d'appel aux api
        """
        # print(params)
        params['autocomplete'] = 0
        params['q'] = params['q'].strip()
        params['limit'] = 1
        if depcom != "" and depcom is not None and re.match(r'^[0-9]{5}$', depcom):
            params['citycode'] = depcom
        try:
            r = s[api].get(api, params=params)
            j = json.loads(r.text)
            # print(j)
            if 'features' in j and len(j['features']) > 0:
                j['features'][0]['l4'] = l4
                j['features'][0]['geo_l4'] = ''
                j['features'][0]['geo_l5'] = ''
                if api != addok_url['poi']:
                    # regénération lignes 4 et 5 normalisées
                    name = j['features'][0]['properties']['name']
                    ligne4 = re.sub(r'\(.*$', '', name).strip()
                    ligne4 = re.sub(r',.*$', '', ligne4).strip()
                    ligne5 = ''
                    j['features'][0]['geo_l4'] = ligne4
                    if '(' in name:
                        ligne5 = re.sub(r'.*\((.*)\)', r'\1', name).strip()
                        j['features'][0]['geo_l5'] = ligne5
                    if ',' in name:
                        ligne5 = re.sub(r'.*,(.*)', r'\1', name).strip()
                        j['features'][0]['geo_l5'] = ligne5
                    # ligne 4 et 5 identiques ? on supprime la 5
                    if j['features'][0]['geo_l5'] == j['features'][0]['geo_l4']:
                        j['features'][0]['geo_l5'] = ''
                return(j['features'][0])
            else:
                return(None)
        except Exception as e:
            logging.error(f"Error contacting geoapi {api} : {e.__class__.__name__} - {str(e)} - {r} - {r.text}")
            # print(json.dumps({'action': 'erreur', 'api': api, 'params': params, 'l4': l4},ensure_ascii=False))
            return(None)

    def trace(txt):
        if False:
            print(txt)

    #def des csv
    file= "file_name"

    # ---------------------------- chargement de la liste des communes et lat/lon
    commune_insee = {}
    communes = csv.DictReader(open(os.path.join(path_to_geodata, 'geo_cog_new_2018.csv'), 'r',encoding='utf8'), delimiter=',')
    #['depcom','cp','ncc','nccenr','pole','longitude', 'latitude', 'geo_score', 'geo_type','geo_city']
    for commune in communes:
        if commune['longitude']=="":
            commune_insee[commune['depcom']]=[commune['nccenr'],"","",commune['cp']]
        else:
            commune_insee[commune['depcom']]=[commune['nccenr'],round(float(commune['latitude']),6),round(float(commune['longitude']),6),commune['cp']]
    #Table de suivi des modifications de depcom
    #['depcom','ncc','nccenr','depcom_pole']
    commune_old_insee = {}
    communes_old = csv.DictReader(open(os.path.join(path_to_geodata, 'geo_cog_old_2018.csv'), 'r',encoding='utf8'), delimiter=',')
    for commune_old in communes_old:
        commune_old_insee[commune_old['depcom']]=[commune_old['nccenr'],commune_old['depcom_pole']]
    #-----------------------------------------------------------------------------------

    #Debut des affaires
    geocode_count = 0
    ok = 0
    total = 0

    numbers = re.compile('(^[0-9]*)')
    ccial = r'((C|CTRE|CENTRE|CNTRE|CENT|ESPACE) (CCIAL|CIAL|COM|COMM|COMMERC|COMMERCIAL)|CCR|C\.CIAL|C\.C|CCIAL|CC)'

    stats = {'action': 'progress', 'housenumber': 0, 'interpolation': 0,
             'street': 0, 'locality': 0, 'municipality': 0, 'vide': 0,
             'townhall': 0, 'poi': 0, 'fichier': file}

    df = pd.read_csv(input_file, encoding='utf-8')
    df = df.fillna('')
    with open(os.path.join(path_to_geodata, 'geo-output'+file), 'w',newline='') as f_out:
        file_geo = csv.writer(f_out,delimiter=";")
        columns_geo = ['cabbi','longitude', 'latitude', 'geo_score', 'geo_type', 'geo_adresse']
        df_result = pd.DataFrame(columns=columns_geo)
        file_geo.writerow(['cabbi','longitude', 'latitude', 'geo_score', 'geo_type', 'geo_adresse'])

        for i, et in df.iterrows():
            total = total + 1

            # mapping des champs de file_in
            cabbi=et.cabbi
            #  
            # au cas où numvoie contiendrait autre chose que des chiffres...
            numvoie = numbers.match(str(et.numvoi_x) if et.numvoi_x is not None else "").group(0)
            indrep = et.bister_x if et.bister_x is not None else ""
            typvoie = et.typevoi_x if et.typevoi_x is not None else ""
            libvoie = et.nomvoi_x if et.nomvoi_x is not None else ""
            compladr = et.cpladr_x if et.cpladr_x is not None else ""
            # lecture code INSEE/nom de la commune
            depcom = str(et.clt_c_c) if et.clt_c_c is not None else ""
            if depcom != "" and depcom is not None and re.match(r'^[0-9]{5}$', depcom):
            # correction depcom
                if depcom not in commune_insee:
                    depcom=commune_old_insee[depcom][1]
                ville=commune_insee[depcom][0]
                cp=commune_insee[depcom][3]
                #ville = et[47]

                #Travail du numvoie
                if numvoie == '' and numbers.match(libvoie).group(0):
                    numvoie = numbers.match(libvoie).group(0)
                    libvoie = libvoie[len(numvoie):]
                #Travail du typvoie

                typ_abrege = {
                            'ALL':'Allée',
                            'AV':'Avenue',
                            'BD':'Boulevard',
                            'CAR':'Carrefour',
                            'CD':'Chemin départemental',
                            'CHE':'Chemin',
                            'CHS':'Chaussée',
                            'CITE':'Cité',
                            'CIT':'Cité',
                            'COR':'Corniche',
                            'CRS':'Cours',
                            'CR':'Chemin rural',
                            'DOM':'Domaine',
                            'DSC':'Descente',
                            'ECA':'Ecart',
                            'ESP':'Esplanade',
                            'FG':'Faubourg',
                            'GR':'Grande Rue',
                            'HAM':'Hameau',
                            'HLE':'Halle',
                            'IMP':'Impasse',
                            'LD':'Lieu-dit',
                            'LOT':'Lotissement',
                            'MAR':'Marché',
                            'MTE':'Montée',
                            'PAS':'Passage',
                            'PLN':'Plaine',
                            'PLT':'Plateau',
                            'PL':'Place',
                            'PRO':'Promenade',
                            'PRV':'Parvis',
                            'QUAI':'Quai',
                            'QUA':'Quartier',
                            'QU':'Quai',
                            'RES':'Résidence',
                            'RLE':'Ruelle',
                            'ROC':'Rocade',
                            'RPT':'Rond-point',
                            'RTE':'Route',
                            'RN':'Route nationale',
                            'R':'Rue',
                            'RUE':'Rue',
                            'SEN':'Sentier',
                            'SQ':'Square',
                            'TPL':'Terre-plein',
                            'TRA':'Traverse',
                            'VC':'Chemin vicinal',
                            'VLA':'Villa',
                            'VLGE':'Village'
                            }
                if typvoie in typ_abrege:
                    typvoie = typ_abrege[typvoie]

                #Travail sur le libvoie
                #libvoie = re.sub(r'^PRO ', 'PROMENADE ', libvoie)
                #libvoie = re.sub(r'^LD ', '', libvoie)
                #libvoie = re.sub(r'^LIEU(.|)DIT ', '', libvoie)
                libvoie = re.sub(r'^ADRESSE INCOMPLETE.*', '', libvoie)
                libvoie = re.sub(r'^SANS DOMICILE FIXE', '', libvoie)
                libvoie = re.sub(r'^COMMUNE DE RATTACHEMENT', '', libvoie)

                ligne4G = ('%s%s %s %s %s %s' % (numvoie, indrep, typvoie, libvoie, cp, ville)).strip()
                # print(f"ligne4g is {ligne4G}")
                ligne4D = ('%s%s %s %s %s %s %s' % (numvoie, indrep, typvoie, libvoie, compladr, cp, ville)).strip()

                trace('%s' % (ligne4G))

                # géocodage BAN (ligne4 géo, déclarée ou normalisée si pas trouvé
                # ou score insuffisant)
                ban = None
                if ligne4G != '':
                    geocode_count += 1
                    ban = geocode(addok_url['ban'], {'q': ligne4G}, 'BANG')
                if ban is None or ban['properties']['score'] < score_min and ligne4D != ligne4G and ligne4D != '':
                    geocode_count += 1
                    ban = geocode(addok_url['ban'], {'q': ligne4D}, 'BAND')
                # géocodage BANO (ligne4 géo, déclarée ou normalisée si pas trouvé
                # ou score insuffisant)
                bano = None
                if ban is None or ban['properties']['score'] < 0.9:
                    if ligne4G != '':
                        geocode_count += 1
                        bano = geocode(addok_url['bano'], {'q': ligne4G}, 'BANOG')
                    if bano is None or bano['properties']['score'] < score_min and ligne4D != ligne4G and ligne4D != '':
                        geocode_count += 1
                        bano = geocode(addok_url['bano'], {'q': ligne4D}, 'BANOD')

                if ban is not None:
                    ban_score = ban['properties']['score']
                    trace(ban_score)
                    ban_type = ban['properties']['type']
                    if ['village', 'town', 'city'].count(ban_type) > 0:
                        ban_type = 'municipality'
                else:
                    ban_score = 0
                    ban_type = ''

                if bano is not None:
                    bano_score = bano['properties']['score']
                    trace(bano_score)
                    if bano['properties']['type'] == 'place':
                        bano['properties']['type'] = 'locality'
                    bano['properties']['id'] = 'BANO_'+bano['properties']['id']
                    if bano['properties']['type'] == 'housenumber':
                        bano['properties']['id'] = '%s_%s' % (bano['properties']['id'],  bano['properties']['housenumber'])
                    bano_type = bano['properties']['type']
                    if ['village', 'town', 'city'].count(bano_type) > 0:
                        bano_type = 'municipality'
                else:
                    bano_score = 0
                    bano_type = ''

                # choix de la source
                source = None
                score = 0

                # on a un numéro... on cherche dessus
                if numvoie != '':
                    # numéro trouvé dans les deux bases, on prend BAN
                    # sauf si score inférieur de 20% à BANO
                    if ban_type == 'housenumber' and bano_type == 'housenumber' and ban_score > score_min and ban_score >= bano_score/1.2:
                        source = ban
                        score = ban['properties']['score']
                    elif ban_type == 'housenumber' and ban_score > score_min:
                        source = ban
                        score = ban['properties']['score']
                    elif bano_type == 'housenumber' and bano_score > score_min:
                        source = bano
                        score = bano['properties']['score']
                    # on cherche une interpollation dans BAN
                    elif ban is None or ban_type == 'street' and int(numvoie) > 2:
                        geocode_count += 2
                        ban_avant = geocode(addok_url['ban'], {'q': '%s %s %s %s' % (int(numvoie)-2, typvoie, libvoie, ville)}, 'BANI')
                        ban_apres = geocode(addok_url['ban'], {'q': '%s %s %s %s' % (int(numvoie)+2, typvoie, libvoie, ville)}, 'BANI')
                        if ban_avant is not None and ban_apres is not None:
                            if ban_avant['properties']['type'] == 'housenumber' and ban_apres['properties']['type'] == 'housenumber' and ban_avant['properties']['score'] > 0.5 and ban_apres['properties']['score'] > score_min :
                                source = ban_avant
                                score = ban_avant['properties']['score']/2
                                source['geometry']['coordinates'][0] = round((ban_avant['geometry']['coordinates'][0]+ban_apres['geometry']['coordinates'][0])/2,6)
                                source['geometry']['coordinates'][1] = round((ban_avant['geometry']['coordinates'][1]+ban_apres['geometry']['coordinates'][1])/2,6)
                                source['properties']['score'] = (ban_avant['properties']['score']+ban_apres['properties']['score'])/2
                                source['properties']['type'] = 'interpolation'
                                source['properties']['id'] = ''
                                source['properties']['label'] = numvoie + ban_avant['properties']['label'][len(ban_avant['properties']['housenumber']):]

                # on essaye sans l'indice de répétition (BIS, TER qui ne correspond pas ou qui manque en base)
                if source is None and ban is None and indrep != '':
                    trace('supp. indrep BAN : %s %s %s' % (numvoie, typvoie, libvoie))
                    geocode_count += 1
                    addok = geocode(addok_url['ban'], {'q': '%s %s %s %s' % (numvoie, typvoie, libvoie, ville)}, 'BANR')
                    if addok is not None and addok['properties']['type'] == 'housenumber' and addok['properties']['score'] > score_min:
                        addok['properties']['type'] = 'interpolation'
                        source = addok
                        trace('+ ban  L4G-indrep')
                if source is None and bano is None and indrep != '':
                    trace('supp. indrep BANO: %s %s %s' % (numvoie, typvoie, libvoie))
                    geocode_count += 1
                    addok = geocode(addok_url['bano'], {'q': '%s %s %s %s' % (numvoie, typvoie, libvoie, ville)}, 'BANOR')  
                    if addok is not None and addok['properties']['type'] == 'housenumber' and addok['properties']['score'] > score_min:
                        addok['properties']['type'] = 'interpolation'
                        source = addok
                        trace('+ bano L4G-indrep')

                # pas trouvé ? on cherche une rue
                if source is None and typvoie != '':
                    if ban_type == 'street' and bano_type == 'street' and ban_score > score_min and ban_score >= bano_score/1.2:
                        source = ban
                        score = ban['properties']['score']
                    elif ban_type == 'street' and ban_score > score_min:
                        source = ban
                    elif bano_type == 'street' and bano_score > score_min:
                        source = bano

                # pas trouvé ? on cherche sans numvoie
                if source is None and numvoie != '':
                    trace('supp. numvoie : %s %s %s' % (numvoie, typvoie, libvoie))
                    geocode_count += 1
                    addok = geocode(addok_url['ban'], {'q': '%s %s %s' % (typvoie, libvoie, ville)}, 'BANN')
                    if addok is not None and addok['properties']['type'] == 'street' and addok['properties']['score'] > score_min:
                        source = addok
                        trace('+ ban  L4G-numvoie')
                if source is None and numvoie != '':
                    geocode_count += 1
                    addok = geocode(addok_url['bano'], {'q': '%s %s %s' % (typvoie, libvoie, ville)}, 'BANON')
                    if addok is not None and addok['properties']['type'] == 'street' and addok['properties']['score'] > score_min:
                        source = addok
                        trace('+ bano L4G-numvoie')

                # toujours pas trouvé ? tout type accepté...
                if source is None:
                    if ban_score > score_min and ban_score >= bano_score*0.8:
                        source = ban
                    elif ban_score > score_min:
                        source = ban
                    elif bano_score > score_min:
                        source = bano

                # vraiment toujours pas trouvé comme adresse ?
                # on cherche dans les POI OpenStreetMap...
                if source is None:
                    # Mairies et Hôtels de Ville...
                    if (['MAIRIE','LA MAIRIE','HOTEL DE VILLE'].count(libvoie) > 0) or (['MAIRIE','LA MAIRIE','HOTEL DE VILLE'].count(compladr) > 0):
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': 'hotel de ville', 'poi': 'townhall'}, 'POI')
                        if poi is not None and poi['properties']['score'] > score_min:
                            source = poi
                    # Gares...
                    elif (['GARE', 'GARE SNCF', 'LA GARE'].count(libvoie) > 0) or (['GARE', 'GARE SNCF', 'LA GARE'].count(compladr) > 0):
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': 'gare', 'poi': 'station'}, 'POI')
                        if poi is not None and poi['properties']['score'] > score_min:
                            source = poi
                    # Centres commerciaux...
                    elif re.match(ccial, libvoie) is not None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': re.sub(ccial, '\1 Galerie Marchande', libvoie), 'poi': 'mall'}, 'POI')
                        if poi is not None and poi['properties']['score'] > 0.5:
                            source = poi
                    elif re.match(ccial, compladr) is not None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': re.sub(ccial, '\1 Galerie Marchande', compladr), 'poi': 'mall'}, 'POI')
                        if poi is not None and poi['properties']['score'] > 0.5:
                            source = poi
                    elif re.match(ccial,libvoie) is not None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': re.sub(ccial, '\1 Centre Commercial', libvoie)}, 'POI')
                        if poi is not None and poi['properties']['score'] > 0.5:
                            source = poi
                    elif re.match(ccial,compladr) is not None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': re.sub(ccial, '\1 Centre Commercial', compladr)}, 'POI')
                        if poi is not None and poi['properties']['score'] > 0.5:
                            source = poi
                    # Aéroports et aérodromes...
                    elif re.match(r'(AEROPORT|AERODROME)', libvoie) is not None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': libvoie, 'poi': 'aerodrome'}, 'POI')
                        if poi is not None and poi['properties']['score'] > score_min:
                            source = poi
                    elif re.match(r'(AEROGARE|TERMINAL)', libvoie) is not None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': re.sub(r'(AEROGARE|TERMINAL)', '', libvoie)+' terminal', 'poi': 'terminal'}, 'POI')
                        if poi is not None and poi['properties']['score'] > score_min:
                            source = poi
                    elif re.match(r'(AEROPORT|AERODROME)', compladr) is not None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': compladr, 'poi': 'aerodrome'}, 'POI')
                        if poi is not None and poi['properties']['score'] > score_min:
                            source = poi
                    elif re.match(r'(AEROGARE|TERMINAL)', compladr) is not None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': re.sub(r'(AEROGARE|TERMINAL)', '', compladr)+' terminal', 'poi': 'terminal'}, 'POI')
                        if poi is not None and poi['properties']['score'] > score_min:
                            source = poi

                    # recherche tout type de POI à partir du type et libellé de voie
                    if source is None:
                        geocode_count += 1
                        poi = geocode(addok_url['poi'], {'q': typvoie+' '+libvoie+' '+compladr}, 'POI')
                        if poi is not None and poi['properties']['score'] > 0.7:
                            source = poi

                    if source is not None:
                        if source['properties']['poi'] != 'yes':
                            source['properties']['type'] = source['properties']['type']+'.'+source['properties']['poi']
                            # print(json.dumps({'action': 'poi', 'adr_insee': depcom,
                            #                     'adr_texte': libvoie, 'poi': source},
                            #                    sort_keys=True,ensure_ascii=False))

                if source is not None and score == 0:
                    score = source['properties']['score']

                if source is None:
                    # attention latitude et longitude sont inversées dans le fichier CSV
                    row = [cabbi,'', '', 0, '', '']
                    try:
                        row = [cabbi,commune_insee[depcom][2],commune_insee[depcom][1], 0,'municipality', '']
                        if ligne4G.strip() != '':
                            if typvoie == '' and ((['CHEF LIEU', 'CHEF-LIEU',
                                                  'LE CHEF LIEU', 'LE CHEF-LIEU',
                                                  'BOURG', 'LE BOURG', 'AU BOURG',
                                                  'VILLAGE', 'AU VILLAGE',
                                                  'LE VILLAGE'].count(libvoie) > 0)
                                                   or
                                                  (['CHEF LIEU', 'CHEF-LIEU',
                                                  'LE CHEF LIEU', 'LE CHEF-LIEU',
                                                  'BOURG', 'LE BOURG', 'AU BOURG',
                                                  'VILLAGE', 'AU VILLAGE',
                                                  'LE VILLAGE'].count(libvoie) > 0)):
                                stats['locality'] += 1
                                ok += 1
                            else:
                                stats['municipality'] += 1
                                # print(json.dumps({'action': 'manque',
                                #                  'cabbi': et[0],
                                #                   'adr_comm_insee': depcom,
                                #                   'adr_short': ligne4G.strip(),
                                #                   'adr_long': ligne4D.strip()},
                                #                  sort_keys=True,ensure_ascii=False))
                        else:
                            stats['vide'] += 1
                            ok += 1
                    except:
                        pass
                    file_geo.writerow(row)
                    df_row = pd.DataFrame([row], columns=columns_geo)
                    df_result = df_result.append(df_row, ignore_index=True)
                else:
                    ok += 1
                    if ['village', 'town', 'city'].count(source['properties']['type']) > 0:
                        source['properties']['type'] = 'municipality'
                    stats[re.sub(r'\..*$', '', source['properties']['type'])] += 1
                    
                    file_geo.writerow([cabbi,source['geometry']['coordinates'][0],
                                            source['geometry']['coordinates'][1],
                                            round(source['properties']['score'], 2),
                                            source['properties']['type'],
                                            source['properties']['label']])
                    
                    df_row = pd.DataFrame([[cabbi,source['geometry']['coordinates'][0],
                                            source['geometry']['coordinates'][1],
                                            round(source['properties']['score'], 2),
                                            source['properties']['type'],
                                            source['properties']['label']]],columns=columns_geo)
                    df_result = df_result.append(df_row, ignore_index=True)
            else:
                #pas de depcom donc on n'écrit rien :-)
                file_geo.writerow([cabbi,'','',0,'',''])
                df_row = pd.DataFrame([[cabbi,'','',0,'','']],columns=columns_geo)
                df_result = df_result.append(df_row, ignore_index=True)

    stats['count'] = total
    stats['geocode_count'] = geocode_count
    stats['action'] = 'final'
    if total!=0:
        stats['efficacite'] = round(100*ok/total, 2)
    else:
        stats['efficacite'] = 'NA'
    # print(json.dumps(stats, sort_keys=True,ensure_ascii=False))
    
    return df_result
    
def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    keep_path = os.path.join(dir_path, "geocodage")
    file_to_keep = ["geo_cog_new_2018.csv", "geo_cog_old_2018.csv" ]
    my_driver = bdd.PostGre_SQL_DB()
    my_driver_es = elastic.ElasticDriver()
    
    sql = "SELECT * FROM rp_final_2019 LIMIT 5"
    data = my_driver.read_from_sql(sql)
    delete_all_files_execpt_from_directory(keep_path, file_to_keep)
    data.to_csv(os.path.join(keep_path, "data.csv"),index=False,sep=',')
    df = geo(os.path.join(keep_path, "data.csv"), path_to_geodata=keep_path)
    delete_all_files_execpt_from_directory(keep_path, file_to_keep)

    
if __name__ == '__main__':
    main()