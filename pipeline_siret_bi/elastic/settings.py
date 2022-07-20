{
    "mappings": {
        "properties" : {
          "actet_x_e" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "analyzer" : "stemming"
              }
            }
          },
          "adr_depcom" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_cedex" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_compl" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_distsp" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_l1" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "analyzer" : "stemming"
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_adr"
            ]
          },
          "adr_et_l2" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_adr"
            ]
          },
          "adr_et_l3" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_adr"
            ]
          },
          "adr_et_l4" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_l5" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_l6" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_l7" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_loc_geo" : {
            "type" : "text",
            "fields" : {
              "depcom" : {
                "type" : "text",
                "analyzer" : "trunc5"
              },
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_post" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_voie_lib" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_voie_num" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_voie_repet" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "adr_et_voie_type" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "ape" : {
            "type" : "text",
            "fields" : {
              "2car" : {
                "type" : "text",
                "analyzer" : "trunc2"
              },
              "3car" : {
                "type" : "text",
                "analyzer" : "trunc3"
              },
              "4car" : {
                "type" : "text",
                "analyzer" : "trunc4"
              },
              "5car" : {
                "type" : "text",
                "analyzer" : "trunc5"
              },
              "div_naf" : {
                "type" : "text",
                "analyzer" : "trunc2"
              },
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              }
            }
          },
          "apet" : {
            "type" : "text",
            "fields" : {
              "2car" : {
                "type" : "text",
                "analyzer" : "trunc2"
              },
              "3car" : {
                "type" : "text",
                "analyzer" : "trunc3"
              },
              "4car" : {
                "type" : "text",
                "analyzer" : "trunc4"
              },
              "5car" : {
                "type" : "text",
                "analyzer" : "trunc5"
              },
              "div_naf" : {
                "type" : "text",
                "analyzer" : "trunc2"
              },
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              }
            }
          },
          "cj" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "denom" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "acro" : {
                "type" : "text",
                "analyzer" : "acronym",
                "search_analyzer" : "simple",
                "fielddata" : True
              },
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "analyzer" : "stemming"
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_denom"
            ],
            "fielddata" : True
          },
          "denom_condense" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "norms" : False,
                "analyzer" : "stemming"
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_denom"
            ]
          },
          "creat_daaaammjj" : {
            "type" : "integer",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "eff_3112_et" : {
            "type" : "integer",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "eff_3112_uniteLegale" : {
            "type" : "integer",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "eff_effet_daaaammjj_uniteLegale" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "eff_et_effet_daaaammjj" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "eff_etp_et" : {
            "type" : "integer",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "eff_etp_uniteLegale" : {
            "type" : "integer",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "enquete_re" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "enquete_re_groupe" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "enseigne" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_denom"
            ]
          },
          "enseigne_et1" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_denom"
            ]
          },
          "enseigne_re" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "analyzer" : "stemming"
              }
            },
            "fielddata" : True
          },
          "enseigne_re_groupe" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "analyzer" : "stemming"
              }
            },
            "fielddata" : True
          },
          "fourretout" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "norms" : False,
                "analyzer" : "stemming",
                "search_analyzer" : "search_syn_rs"
              }
            }
          },
          "geo_adresse" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "geo_id" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "geo_l4" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "geo_l5" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "geo_ligne" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "geo_score" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "geo_type" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "latitude" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "location" : {
            "type" : "geo_point",
            "ignore_malformed" : True
          },
          "longitude" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "n_bi_2013" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "n_bi_2014" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "n_bi_2015" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "n_bi_2016" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "nic" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "nic_siege" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "nom_comm_et" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "acro" : {
                "type" : "text",
                "analyzer" : "acronym",
                "search_analyzer" : "simple",
                "fielddata" : True
              },
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "norms" : False,
                "analyzer" : "stemming"
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_denom"
            ]
          },
          "profs_x_e" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "analyzer" : "stemming"
              }
            }
          },
          "qual" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "region" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "region_impl" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "region_mult" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "rs_adr" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "norms" : False,
                "analyzer" : "stemming",
                "search_analyzer" : "search_syn_rs"
              }
            }
          },
          "rs_denom" : {
            "type" : "text",
            "norms" : False,
            "fields" : {
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "norms" : False,
                "analyzer" : "stemming",
                "search_analyzer" : "search_syn_rs"
              }
            }
          },
          "rs_x_e" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              },
              "stem" : {
                "type" : "text",
                "analyzer" : "stemming"
              }
            }
          },
          "sigle" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              },
              "ngr" : {
                "type" : "text",
                "analyzer" : "ngram_analyzer"
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_denom"
            ],
            "analyzer" : "sigle"
          },
          "sigle_denom" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_denom"
            ],
            "analyzer" : "sigle"
          },
          "sigle_l1" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            },
            "copy_to" : [
              "fourretout",
              "rs_denom"
            ],
            "analyzer" : "sigle"
          },
          "sir_adr_et_com_lib" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "sirus_id" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "sourcexyw" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "tr_eff_etp" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "unite_type" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "x" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          },
          "y" : {
            "type" : "text",
            "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
          }
        }
    },
    "settings": {
      "index" : {
        "refresh_interval" : "5s",
        "number_of_shards" : "1",
        "analysis" : {
          "filter" : {
            "french_stop" : {
              "type" : "stop",
              "stopwords" : "_french_"
            },
            "acronymizer" : {
              "pattern" : """(?<=\w)[A-z]*(\s|\b|\-)""",
              "replace" : "",
              "type" : "pattern_replace"
            },
            "french_elision" : {
              "type" : "elision",
              "articles" : [
                "l",
                "m",
                "t",
                "qu",
                "n",
                "s",
                "j",
                "d",
                "c"
              ],
              "articles_case" : "True"
            },
            "trunc5" : {
              "length" : "5",
              "type" : "truncate"
            },
            "trunc4" : {
              "length" : "4",
              "type" : "truncate"
            },
            "trunc3" : {
              "length" : "3",
              "type" : "truncate"
            },
            "trunc2" : {
              "length" : "2",
              "type" : "truncate"
            },
            "cleansigle" : {
              "pattern" : """\.|\-|\s*""",
              "replace" : "",
              "type" : "pattern_replace"
            },
            "synonyms_rs" : {
              "type" : "synonym_graph",
              "synonyms_path" : "analysis/synonymes.txt"
            },
            "french_stemmer" : {
              "type" : "stemmer",
              "language" : "light_french"
            },
            "pattern_stop" : {
              "pattern" : """\ble\b|\bla\b|\bles\b|\bde\b|\bdes\b|\bdu\b|\ben\b|l\'|d\'""",
              "replace" : "",
              "type" : "pattern_replace"
            },
            "toklen3" : {
              "type" : "length",
              "min" : "3"
            },
            "french_keywords" : {
              "keywords" : [
                "Exemple"
              ],
              "type" : "keyword_marker"
            }
          },
          "analyzer" : {
            "ngram_analyzer" : {
              "filter" : "lowercase",
              "tokenizer" : "ngram_tokenizer"
            },
            "acronym" : {
              "filter" : [
                "asciifolding",
                "lowercase",
                "pattern_stop",
                "acronymizer",
                "cleansigle",
                "toklen3"
              ],
              "type" : "custom",
              "tokenizer" : "keyword"
            },
            "search_syn_rs" : {
              "filter" : [
                "asciifolding",
                "french_elision",
                "lowercase",
                "synonyms_rs",
                "french_stop",
                "french_stemmer"
              ],
              "tokenizer" : "letter"
            },
            "trunc5" : {
              "filter" : "trunc5",
              "tokenizer" : "keyword"
            },
            "trunc4" : {
              "filter" : "trunc4",
              "tokenizer" : "keyword"
            },
            "stemming" : {
              "filter" : [
                "asciifolding",
                "french_elision",
                "lowercase",
                "french_stop",
                "french_stemmer",
                "unique"
              ],
              "tokenizer" : "letter"
            },
            "trunc3" : {
              "filter" : "trunc3",
              "tokenizer" : "keyword"
            },
            "trunc2" : {
              "filter" : "trunc2",
              "tokenizer" : "keyword"
            },
            "sigle" : {
              "filter" : [
                "classic"
              ],
              "type" : "custom",
              "tokenizer" : "keyword"
            }
          },
          "tokenizer" : {
            "ngram_tokenizer" : {
              "token_chars" : [
                "letter",
                "digit"
              ],
              "min_gram" : "2",
              "type" : "edge_ngram",
              "max_gram" : "8"
            }
          }
        },
        "number_of_replicas" : "2"
      }
    }
  }