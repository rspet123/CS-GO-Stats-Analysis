import operator
from typing import Dict, List, Tuple, Union

import os

import csgo_utils

import pandas as pd

import plotly.express as px

from csgo.parser import DemoParser

import matplotlib.pyplot as plt
import numpy as np

import plotly

import sklearn

import plotly

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import mplcursors

from tqdm import tqdm

#https://www.kaggle.com/naren3256/kmeans-clustering-and-cluster-visualization-in-3d

#https://flashed.gg/posts/reverse-engineering-hltv-rating/

#https://plotly.com/python/3d-scatter-plots/

#0.0073*KAST + 0.3591*KPR + -0.5329*DPR + 0.2372*Impact + 0.0032*ADR + 0.1587 = Rating 2.0

#2.13*KPR + 0.42*Assist per Round -0.41 ≈ Impact
  
demo_directory = "resources"

min_rounds = 16

demo_data_list = []

PLAYER_DICT = {'k': 0, 'd': 0, 'hs': 0, 'tk': 0, 'ek': 0, 'td': 0, 'a': 0, 'fa': 0, 'rd': 0, 'dmg_taken': 0, 'dmg': 0, 'ef': 0, 'eh': 0, 'hsp': 0, 'kd': 0, 'adr': 0, 'kpr': 0, 'es':0, 'awp_kills':0,'ptc':0,'ft':0,'fs':0,'pf':0,'trd':0,'ctrd':0}

for filename in tqdm(os.listdir(demo_directory)):
    print("loading: " + filename)
    try:
        demo_data_list.append(DemoParser(demofile="resources\\" + filename, demo_id=filename, parse_rate=128).parse())
    except Exception:
        print("Not a .dem file, or otherwise corrupt")

kast_calc = {}
player_names = {}
scoreboard = {}
kill_data = {}
for match in demo_data_list:
    for match_round in match["gameRounds"]:
        round_deaths = []
        round_participation = {}
        ct_avg_econ = match_round["ctSpend"] + match_round["ctStartEqVal"]
        t_avg_econ = match_round["tSpend"]+ match_round["tStartEqVal"]
        print("Round #" + str(match_round["roundNum"]) + "------")
        print("Score")
        print(match_round["tTeam"] + " - " + str(match_round["tScore"]))
        print(match_round["ctTeam"] + " - " + str(match_round["ctScore"]))
        for kill in match_round["kills"]:
            if not kill["isSuicide"]:
                if kill["attackerSteamID"] not in player_names.keys():
                    # We need to "Normalize" names here based on steam ID, as the game only gives us tags
                    # and players can have different tags between games, making it hard to track players over multiple games
                    player_names[kill["attackerSteamID"]] = kill["attackerName"]
                    print("Adding " + kill["attackerName"])
                if kill["victimSteamID"] not in player_names.keys():
                    # Again, we're just keeping track of player names, this time for the one who got shot
                    player_names[kill["victimSteamID"]] = kill["victimName"]
                    print("Adding " + kill["victimName"] + " to players")
                    
                if player_names[kill["attackerSteamID"]] not in kill_data.keys():
                    #Add our kill to the list of a players kills
                    kill_round = kill
                    kill_round['round_num'] = match_round["roundNum"]
                    kill_data[player_names[kill["attackerSteamID"]]] = []
                    kill_data[player_names[kill["attackerSteamID"]]].append(kill_round)
                else:
                    kill_data[player_names[kill["attackerSteamID"]]].append(kill_round)
                    
                
                #Track Kills
                if  player_names[kill["attackerSteamID"]] not in scoreboard.keys():
                    scoreboard[player_names[kill["attackerSteamID"]]] = PLAYER_DICT.copy()
                scoreboard[player_names[kill["attackerSteamID"]]]["k"] = scoreboard[player_names[kill["attackerSteamID"]]].get("k",0)+1
                #Track participation for KAST calculation
                round_participation[player_names[kill["attackerSteamID"]]] = 1
                if kill["weapon"] == 'AWP':
                    #Track AWP Kills
                    scoreboard[player_names[kill["attackerSteamID"]]]["awp_kills"] = scoreboard[player_names[kill["attackerSteamID"]]].get("awp_kills",0)+1
                #Track Deaths
                if  player_names[kill["victimSteamID"]] not in scoreboard.keys():
                    scoreboard[player_names[kill["victimSteamID"]]] = PLAYER_DICT.copy()
                scoreboard[player_names[kill["victimSteamID"]]]["d"] = scoreboard[player_names[kill["victimSteamID"]]].get("d",0)+1
                round_deaths.append(player_names[kill["victimSteamID"]])

                #Track Trades
                if kill["isTrade"]:
                    #Add Trade Kill
                    if  player_names[kill["attackerSteamID"]] not in scoreboard.keys():
                        scoreboard[player_names[kill["attackerSteamID"]]] = PLAYER_DICT.copy()
                    scoreboard[player_names[kill["attackerSteamID"]]]["tk"] = scoreboard[player_names[kill["attackerSteamID"]]].get("tk",0)+1
                    
                    #Track Traded Death
                    if kill["playerTradedSteamID"] not in player_names.keys():
                        print("Adding " + kill["playerTradedName"] + " to players")
                        player_names[kill["playerTradedSteamID"]] = kill["playerTradedName"]
                        
                    if player_names[kill["playerTradedSteamID"]] not in scoreboard.keys():
                        scoreboard[player_names[kill["playerTradedSteamID"]]] = PLAYER_DICT.copy()
                    scoreboard[player_names[kill["playerTradedSteamID"]]]["td"] = scoreboard[player_names[kill["playerTradedSteamID"]]].get("td",0)+1
                    #Track participation for KAST calculation
                    round_participation[player_names[kill["playerTradedSteamID"]]] = 1
                
                
                if kill["isFirstKill"]:
                    if kill["attackerSide"] == "T":
                        #Track Entries
                        if  player_names[kill["attackerSteamID"]] not in scoreboard.keys():
                            scoreboard[player_names[kill["attackerSteamID"]]] = PLAYER_DICT.copy()
                        scoreboard[player_names[kill["attackerSteamID"]]]["ek"] = scoreboard[player_names[kill["attackerSteamID"]]].get("ek",0)+1
                    elif kill["attackerSide"] == "CT":
                        #Track Entry Holds / Entry Fails
                        if  player_names[kill["attackerSteamID"]] not in scoreboard.keys():
                            scoreboard[player_names[kill["attackerSteamID"]]] = PLAYER_DICT.copy()
                        scoreboard[player_names[kill["attackerSteamID"]]]["eh"] = scoreboard[player_names[kill["attackerSteamID"]]].get("eh",0)+1
                        scoreboard[player_names[kill["victimSteamID"]]]["ef"] = scoreboard[player_names[kill["victimSteamID"]]].get("ef",0)+1
                #Track Headshots
                if kill["isHeadshot"]:
                    if  player_names[kill["attackerSteamID"]] not in scoreboard.keys():
                        scoreboard[player_names[kill["attackerSteamID"]]] = PLAYER_DICT.copy()
                    scoreboard[player_names[kill["attackerSteamID"]]]["hs"] = scoreboard[player_names[kill["attackerSteamID"]]].get("hs",0)+1
                
                #Track Flash Assists
                if kill["victimBlinded"]:
                    if kill["flashThrowerSide"] == kill["attackerSide"]:
                        if kill["flashThrowerSteamID"] not in player_names.keys():
                            player_names[kill["flashThrowerSteamID"]] = kill["flashThrowerName"]
                        if  player_names[kill["flashThrowerSteamID"]] not in scoreboard.keys():
                            scoreboard[player_names[kill["flashThrowerSteamID"]]] = PLAYER_DICT.copy()
                        scoreboard[player_names[kill["flashThrowerSteamID"]]]["fa"] = scoreboard[player_names[kill["flashThrowerSteamID"]]].get("fa",0)+1
                #Check for assist, process if exists
                if kill["assisterSteamID"] is None:
                    print(kill["attackerName"] + " kills " +kill["victimName"] + " with " + kill["weapon"]) 
                else:
                    if kill["assisterSteamID"] not in player_names.keys():
                        # Keeping track of players still
                        player_names[kill["assisterSteamID"]] = kill["assisterName"]
                    print(kill["attackerName"] + " kills " +kill["victimName"] + " with " + kill["weapon"] + " assisted by " + kill["assisterName"])
                    #Track Assists
                    if  player_names[kill["assisterSteamID"]] not in scoreboard.keys():
                        scoreboard[player_names[kill["assisterSteamID"]]] = PLAYER_DICT.copy()
                    scoreboard[player_names[kill["assisterSteamID"]]]["a"] = scoreboard[player_names[kill["assisterSteamID"]]].get("a",0)+1
                    #Track participation for KAST calculation
                    round_participation[player_names[kill["assisterSteamID"]]] = 1
        #Track Rounds played per Player
        if match_round.get("frames",False):
        #T Players
            for t_player in match_round["frames"][0]["t"]["players"]:
                if t_player["steamID"] not in player_names.keys():
                    player_names[t_player["steamID"]] = t_player["name"]
                if player_names[t_player["steamID"]] not in scoreboard.keys():
                    scoreboard[player_names[t_player["steamID"]]] = PLAYER_DICT.copy()
                scoreboard[player_names[t_player["steamID"]]]["trd"] = scoreboard[player_names[t_player["steamID"]]].get("trd",0)+1
                scoreboard[player_names[t_player["steamID"]]]["rd"] = scoreboard[player_names[t_player["steamID"]]].get("rd",0)+1
                scoreboard[player_names[t_player["steamID"]]]["team"] = t_player["team"]
                # KAST Calculation for Survival
                if player_names[t_player["steamID"]] not in round_deaths:
                    round_participation[player_names[t_player["steamID"]]] = 1
            #CT Players
            for ct_player in match_round["frames"][0]["ct"]["players"]:
                if ct_player["steamID"] not in player_names.keys():
                    player_names[ct_player["steamID"]] = ct_player["name"]
                if player_names[ct_player["steamID"]] not in scoreboard.keys():
                    scoreboard[player_names[ct_player["steamID"]]] = PLAYER_DICT.copy()
                scoreboard[player_names[ct_player["steamID"]]]["ctrd"] = scoreboard[player_names[ct_player["steamID"]]].get("ctrd",0)+1
                scoreboard[player_names[ct_player["steamID"]]]["rd"] = scoreboard[player_names[ct_player["steamID"]]].get("rd",0)+1
                scoreboard[player_names[ct_player["steamID"]]]["team"] = ct_player["team"]
            # KAST Calculation for Survival
            if player_names[ct_player["steamID"]] not in round_deaths:
                round_participation[player_names[ct_player["steamID"]]] = 1
                
            for player in round_participation.keys():
                scoreboard[player]["ptc"] = scoreboard[player].get("ptc",0)+1
                
        
        #Track Damage
        for damage_event in match_round["damages"]:
            if not damage_event["isFriendlyFire"]:
                if damage_event["attackerSteamID"] and damage_event["victimSteamID"]:
                    scoreboard[player_names[damage_event["attackerSteamID"]]]["dmg"] = scoreboard[player_names[damage_event["attackerSteamID"]]].get("dmg",0)+damage_event["hpDamage"]
                    scoreboard[player_names[damage_event["victimSteamID"]]]["dmg_taken"] = scoreboard[player_names[damage_event["victimSteamID"]]].get("dmg_taken",0)+damage_event["hpDamage"]
        #Track Grenades
        for nade in match_round["grenades"]:
            #Track Flashes thrown for flash efficiency
            if nade["grenadeType"] == "Flashbang":
                scoreboard[player_names[nade["throwerSteamID"]]]["ft"] = scoreboard[player_names[nade["throwerSteamID"]]].get('ft',0)+1
        
        for flash in match_round["flashes"]:
            #Check for teamflashes
            if not flash["attackerSide"] == flash["playerSide"]:
                if flash["attackerSteamID"] not in player_names.keys():
                    player_names[flash["attackerSteamID"]] = flash["attackerName"]
                scoreboard[player_names[flash["attackerSteamID"]]]["fs"] = scoreboard[player_names[flash["attackerSteamID"]]].get("fs",0) + flash["flashDuration"]
                scoreboard[player_names[flash["attackerSteamID"]]]["pf"] = scoreboard[player_names[flash["attackerSteamID"]]].get("pf",0) + 1
        
        if match_round["winningTeam"]:
            print(match_round["winningTeam"] + " wins!")
players_to_delete = []
for player in scoreboard.keys():
    # remove players with less than min rounds
    if scoreboard[player]["rd"] < min_rounds:
        print("Deleting " + player + ", only has " + str(scoreboard[player]["rd"]) + " rounds played")
        players_to_delete.append(player)
        continue
    # Headshot %
    try:
        scoreboard[player]["hsp"] = float(scoreboard[player]["hs"]) / float(scoreboard[player]["k"])
    except ZeroDivisionError:
        print(player + " never got a kill, sad")
        scoreboard[player]["hsp"] = 0
    # Classic Kill/Death Ratio
    try:
        scoreboard[player]["kd"] = float(scoreboard[player]["k"]) / float(scoreboard[player]["d"])
    except ZeroDivisionError:
        print(player + " never died, wow")
        scoreboard[player]["kd"] = float(scoreboard[player]["k"])
    # Average Damage per Round
    scoreboard[player]["adr"] = float(scoreboard[player]["dmg"]) / float(scoreboard[player]["rd"])
    # Kills per Round
    scoreboard[player]["kpr"] = float(scoreboard[player]["k"]) / float(scoreboard[player]["rd"])
    # Deaths per Round
    scoreboard[player]["dpr"] = float(scoreboard[player]["d"]) / float(scoreboard[player]["rd"])
    # Entry Success Rate
    try:
        scoreboard[player]["es"] = float(scoreboard[player]["ek"]) / (float(scoreboard[player].get("ek",0))+float(scoreboard[player].get("ef",)))
    except ZeroDivisionError:
        print(player + " never tried to entry, what a coward")
        scoreboard[player]["es"] = 0 #this should be like, -1 or something, but that ends up weird on the graph
    # avg entry success per t round 
    scoreboard[player]["espr"] = float(scoreboard[player]["ek"]) / float(scoreboard[player]["trd"])
    # entry attempts
    scoreboard[player]["ea"] = ((scoreboard[player].get("ek",0))+(scoreboard[player].get("ef",0)))
    # entry attempts per t round
    scoreboard[player]["ear"] = float(scoreboard[player]["ea"]) / float(scoreboard[player]["trd"])
    # flash assists per round
    scoreboard[player]["far"] = float(scoreboard[player]["fa"]) / float(scoreboard[player]["rd"])
    # combined assists per round
    scoreboard[player]["car"] = float(scoreboard[player]["fa"] + scoreboard[player]["a"]) / float(scoreboard[player]["rd"])
    # support score per round
    # support score is a new metric that basically describes how much a player assists his fellow players per round
    scoreboard[player]["scr"] = float(scoreboard[player]["fa"] + scoreboard[player]["a"] + scoreboard[player]["tk"]) / float(scoreboard[player]["rd"])
    # awp kills per round
    scoreboard[player]["akr"] = float(scoreboard[player]["awp_kills"]) / float(scoreboard[player]["rd"])
    # survives
    scoreboard[player]["s"] = scoreboard[player]["rd"] - scoreboard[player]["d"]
    try:
        # flash efficiency, players flashed / flash thrown
        scoreboard[player]["pff"] = scoreboard[player]["pf"] / scoreboard[player]["ft"]
        # flash efficiency, seconds blinded / flash thrown
        scoreboard[player]["sff"] = scoreboard[player]["fs"] / scoreboard[player]["ft"]
    except ZeroDivisionError:
        print(player + " never threw a flash, that's dumb")
        scoreboard[player]["sff"] = 0
        scoreboard[player]["pff"] = 0
    # survives per round
    scoreboard[player]["sr"] = scoreboard[player]["s"] / scoreboard[player]["rd"]
    # KAST, Kill/Assist/Survive/Traded per round
    scoreboard[player]["kast"] = (scoreboard[player]["ptc"] / scoreboard[player]["rd"])
    # IMPACT, HLTV stat used for calculating rating
    scoreboard[player]["impact"] = 2.13 * scoreboard[player]["kpr"] + (.42 * (scoreboard[player]["a"]/scoreboard[player]["rd"]))-.41
    # HLTV Rating
    # TODO, seems broken
    scoreboard[player]["hltv"] = ((.0073*scoreboard[player]["kast"]) + (.3591 * scoreboard[player]["kpr"]) + (-0.5329 * scoreboard[player]["dpr"]) + (.2372 * scoreboard[player]["impact"]) + (.0032 * scoreboard[player]["adr"]) + .1587)
    # awp kill percent
    try:
        scoreboard[player]["akp"] = float(scoreboard[player]["awp_kills"]) / float(scoreboard[player]["k"])
    except ZeroDivisionError:
        print(player + " never got a kill, sad")
        scoreboard[player]["akp"] = 0
        
for player in players_to_delete:
    del(scoreboard[player])
df_score = pd.DataFrame.from_dict(scoreboard, orient='index')

fig = px.scatter_3d(df_score, x = "ear",y="akp",z="scr",color = "adr",size="kast",text = df_score.index, symbol = 'team')

fig.show()
fig.write_html("roles.html")
