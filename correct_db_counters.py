from data_import.bdd import PostGre_SQL_DB
from config import *
import logging
import logging.config
from sqlalchemy import text

with open(os.path.join(os.path.dirname(__file__), 'pipeline_siret_bi', 'logging.conf.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)

db = PostGre_SQL_DB(
    db_host,
    db_port,
    db_dbname,
    db_user,
    db_password
)


def extract_lot_counts(lot_df):
    """
    Extract various counts on lots
    """
    nb_bi = len(lot_df)
    nb_bi_rep_act = len(lot_df[lot_df.i_reprise_act.notnull()])
    nb_bi_rep_act_N = len(lot_df[lot_df.i_reprise_act == 'N'])
    nb_bi_rep_act_att = len(lot_df[lot_df.i_reprise_act == 'A'])
    nb_bi_rep_act_traite = len(lot_df[lot_df.i_reprise_act == 'V'])
    nb_bi_rep_prof = len(lot_df[lot_df.i_reprise_prof.notnull()])
    nb_bi_rep_prof_N = len(lot_df[lot_df.i_reprise_prof == 'N'])
    nb_bi_rep_prof_att = len(lot_df[lot_df.i_reprise_prof == 'A'])
    nb_bi_rep_prof_traite = len(lot_df[lot_df.i_reprise_prof == 'V'])
    nb_bi_att = len(lot_df[(lot_df.i_reprise_act == 'A') | (lot_df.i_reprise_prof == 'A')])
    nb_bi_traite = len(lot_df[(lot_df.i_reprise_act == 'V') & (lot_df.i_reprise_prof == 'V')]) \
                    + len(lot_df[(lot_df.i_reprise_act == 'V') & (lot_df.i_reprise_prof.isnull())]) \
                        + len(lot_df[(lot_df.i_reprise_act.isnull()) & (lot_df.i_reprise_prof == 'V')])
    return {
        "nb_bi": nb_bi, 
        "nb_bi_att": nb_bi_att, 
        "nb_bi_traite": nb_bi_traite, 
        "nb_bi_rep_act": nb_bi_rep_act, 
        "nb_bi_rep_act_att": nb_bi_rep_act_att, 
        "nb_bi_rep_act_traite": nb_bi_rep_act_traite,
        "nb_bi_rep_prof": nb_bi_rep_prof, 
        "nb_bi_rep_prof_att": nb_bi_rep_prof_att, 
        "nb_bi_rep_prof_traite": nb_bi_rep_prof_traite
    }

def check_lots(correct_db=False):
    """
    Check all lots. If correct_db, db is updated with correct counts
    """
    counter = 0
    counter_closed = 0
    query = "select * from rp.lots_reprise where lot_statut != 'N' order by lot_id"
    for _, lot in db.read_from_sql(query).iterrows():
        query_lot = f"select * from rp.individus where lot_id = {lot['lot_id']}"
        lot_df = db.read_from_sql(query_lot)
        counts = extract_lot_counts(lot_df)
        lot_counter = 0
        lot_counter_closed = 0
        vars_to_update = {}
        for varname, var in counts.items():
            if lot[varname] != var:
                lot_counter = 1
                logging.debug(f"lot {lot['lot_id']}: incorrect {varname}")
                vars_to_update[varname] = var
        if len(vars_to_update):
            # the data was incorrect
            if counts['nb_bi_traite'] == counts['nb_bi'] and lot['lot_statut'] != 'T':
                lot_counter_closed = 1
                logging.debug(f"lot {lot['lot_id']}: lot_statut must change")
        if correct_db and len(vars_to_update):
            with db.engine.connect() as conn:
                update_str = "UPDATE rp.lots_reprise SET "
                update_str += ", ".join([f"{varname} = {val}" for varname, val in vars_to_update.items()])
                if lot_counter_closed:
                    update_str += ", lot_statut = 'T'"
                update_str += f" WHERE lot_id = {lot['lot_id']}"
                update_query = text(update_str)
                result = conn.execute(update_query)
                logging.debug(f"{result.rowcount} row(s) updated")
        counter += lot_counter
        counter_closed += lot_counter_closed

    logging.info(f'{counter} lots incorrects')
    logging.info(f'{counter_closed} lots have incorrect lot_statut')
    return counter, counter_closed


def extract_vague_counts(vague_df):
    """
    Extract various counts on vagues
    """
    nb_lots = len(vague_df)
    nb_lots_en_cours = len(vague_df[vague_df.lot_statut == "E"])
    nb_lots_trait = len(vague_df[vague_df.lot_statut == 'T'])
    return {
        "nb_lots": nb_lots, 
        "nb_lots_en_cours": nb_lots_en_cours, 
        "nb_lots_trait": nb_lots_trait, 
    }


def check_vagues(correct_db=False):
    """
    Check all vagues. If correct_db, db is updated with correct counts
    """
    counter = 0
    counter_closed = 0
    query = "select * from rp.vagues_reprise order by vague_id"
    for _, vague in db.read_from_sql(query).iterrows():
        query_vague = f"select * from rp.lots_reprise where vague_id = {vague['vague_id']}"
        vague_df = db.read_from_sql(query_vague)
        counts = extract_vague_counts(vague_df)
        vague_counter = 0
        vague_counter_closed = 0
        vars_to_update = {}
        for varname, var in counts.items():
            if vague[varname] != var:
                vague_counter = 1
                logging.debug(f"vague {vague['vague_id']}: incorrect {varname}")
                vars_to_update[varname] = var
        if len(vars_to_update):
            # the data was incorrect
            if counts['nb_lots_trait'] == counts['nb_lots'] and vague['vague_statut'] != 'T':
                vague_counter_closed = 1
                logging.debug(f"vague {vague['vague_id']}: vague_statut must change")
        if correct_db and len(vars_to_update):
            with db.engine.connect() as conn:
                update_str = "UPDATE rp.vagues_reprise SET "
                update_str += ", ".join([f"{varname} = {val}" for varname, val in vars_to_update.items()])
                if vague_counter_closed:
                    update_str += ", vague_statut = 'T'"
                update_str += f" WHERE vague_id = {vague['vague_id']}"
                update_query = text(update_str)
                result = conn.execute(update_query)
                logging.debug(f"{result.rowcount} row(s) updated")
        counter += vague_counter

    logging.info(f'{counter} vagues incorrects')
    logging.info(f'{counter_closed} vagues have incorrect vague_statut')
    return counter, counter_closed


if __name__ == '__main__':
    """
    Ce script vérifie les compteurs (nb de BI à divers états dans un lot, nb de lots à divers état dans une vague)
    Si correct_db = True, la DB est updatée pour remettre les compteurs aux bonnes valeurs.
    Si éligible, le statut est passé de 'E' à 'T'

    """
    check_lots(correct_db=True)
    check_vagues(correct_db=True)