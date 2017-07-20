#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import json
import requests

se_url = "http://maayanlab.net/SEP-L1000/search.php"
auc_url = "http://maayanlab.net/SEP-L1000/se_profile.php"

se_params = {"term": "."}


def get_data(url, params):
    resp = requests.get(url=url, params=params)
    return json.loads(resp.text)


if __name__ == '__main__':
    side_effects = get_data(se_url, se_params)

    f = open("paper_auc.txt", "w")
    f_all = open("paper_auc_all.txt", "w")

    for se in side_effects["se"]:
        if se["umls_id"] is not None:
            auc_params = {"umls_id": se["umls_id"]}
            se_info = get_data(auc_url, auc_params)

            if se_info["auroc"] is not None:
                f.write("%s %s\n" % (se_info["name"].replace(' ', '_'), se_info["auroc"]))

            f_all.write("%s %s\n" % (se_info["name"].replace(' ', '_'), se_info["auroc"]))
            print("%s %s" % (se_info["name"], se_info["auroc"]))

    f.close()
    f_all.close()
