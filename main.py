import os

import pandas as pd

import plotly.express as px

from csgo.parser import DemoParser

import matplotlib.pyplot as plt

import numpy as np

import plotly

from sklearn.cluster import KMeans

import csgo_stat_functions

from tqdm import tqdm

import pickle

#0.0073*KAST + 0.3591*KPR + -0.5329*DPR + 0.2372*Impact + 0.0032*ADR + 0.1587 = Rating 2.0

#2.13*KPR + 0.42*Assist per Round -0.41 ≈ Impact
  
demo_directory = "resources"

min_rounds = 16

demo_data_list = []

pistols = ['USP-S','Glock-18','P250','Five-SeveN','Tec-9','Desert Eagle','R8 Revolver','Dual Berettas']

dmg_nades = ["HE Grenade","Molotov","Incendiary Grenade"]

low_buys = ['Full Eco','Semi Eco']

full_buys = ['Full Buy','Semi Buy']


PLAYER_DICT = {'k': 0, 'd': 0, 'hs': 0, 'tk': 0, 'ek': 0, 'td': 0, 'a': 0, 'fa': 0, 'rd': 0, 'dmg_taken': 0, 'dmg': 0, 'ef': 0, 'eh': 0, 'hsp': 0, 'kd': 0, 'adr': 0, 'kpr': 0, 'es':0, 'awp_kills':0,'ptc':0,'ft':0,'fs':0,'pf':0,'trd':0,'ctrd':0,'dist':0,'fld':0,'flk':0,'flkr':0,"fldr":0,'afk':0,'aafk':0,'ddd':0,'ddist':0,'awdr':0,'awp_deaths':0,'awdd':0,'aduel':0,'sk':0,'skk':0,'sd':0,'sdd':0,'adv':0,'avadv':0,'pk':0,'pkr':0,'sek':0,'asek':0,'ecok':0,'aecok':0,'ecfr':0,'dek':0,'adek':0}
demo_ids = os.listdir(demo_directory)

if not os.path.exists("demos.kat"):
    for filename in tqdm(os.listdir(demo_directory)):
        print("loading: " + filename)
        try:
            demo_data_list.append(DemoParser(demofile="resources\\" + filename, demo_id=filename,outpath="JSONLogs", parse_rate=128).parse())
        except Exception:
            print("Not a .dem file, or otherwise corrupt")
    #Save our demos json file, it's huge and I don't want to reacquire it on startup every time
    pickle.dump( (demo_data_list,len(demo_data_list),demo_ids), open( "demos.kat", "wb" ))
else:
    if not demo_ids == (pickle.load(open( "demos.kat", "rb" )))[2]:
        for filename in tqdm(os.listdir(demo_directory)):
            print("loading: " + filename)
            try:
                demo_data_list.append(DemoParser(demofile="resources\\" + filename, demo_id=filename,outpath="JSONLogs", parse_rate=128).parse())
            except Exception:
                print("Not a .dem file, or otherwise corrupt")
        #Save our demos json file, it's huge and I don't want to reacquire it on startup every time
        pickle.dump( (demo_data_list,len(demo_data_list),demo_ids), open( "demos.kat", "wb" ))
    else:
        demo_data_list = (pickle.load(open("demos.kat", "rb" )))[0]

kast_calc = {}
player_names = {}
scoreboard = {}
kill_data = {}
flank_angle = 45
for match in demo_data_list:
    for match_round in match["gameRounds"]:
        round_deaths = []
        round_participation = {}
        econ = {'CT':match_round["ctBuyType"],"T":match_round["tBuyType"]}
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
                    kill_data[player_names[kill["attackerSteamID"]]] = []
                    kill_data[player_names[kill["attackerSteamID"]]].append(kill)
                else:
                    kill_data[player_names[kill["attackerSteamID"]]].append(kill)
                    
                
                #Track Kills
                if  player_names[kill["attackerSteamID"]] not in scoreboard.keys():
                    scoreboard[player_names[kill["attackerSteamID"]]] = PLAYER_DICT.copy()
                scoreboard[player_names[kill["attackerSteamID"]]]["k"] = scoreboard[player_names[kill["attackerSteamID"]]].get("k",0)+1
                
                
                ############### Economy Calculations###################
                #Track kills in the same econ state
                if econ[kill["attackerSide"]] == econ[kill["victimSide"]]:
                    scoreboard[player_names[kill["attackerSteamID"]]]["sek"] = scoreboard[player_names[kill["attackerSteamID"]]].get("sek",0)+1
                #Track Eco Frags
                if (econ[kill["attackerSide"]] in full_buys) and (econ[kill["victimSide"]] in low_buys):
                    scoreboard[player_names[kill["attackerSteamID"]]]["ecok"] = scoreboard[player_names[kill["attackerSteamID"]]].get("ecok",0)+1
                
                #Track Down Eco Kills
                if (econ[kill["victimSide"]] in full_buys) and (econ[kill["attackerSide"]] in low_buys):
                    scoreboard[player_names[kill["attackerSteamID"]]]["dek"] = scoreboard[player_names[kill["attackerSteamID"]]].get("dek",0)+1
                
                #Track Pistol Kills
                if kill['weapon'] in pistols:
                    print("Pistol Kill")
                    scoreboard[player_names[kill["attackerSteamID"]]]["pk"] = scoreboard[player_names[kill["attackerSteamID"]]].get("pk",0)+1
                
                #Track Average Distance of kill
                scoreboard[player_names[kill["attackerSteamID"]]]["dist"] = scoreboard[player_names[kill["attackerSteamID"]]].get("dist",0)+kill["distance"]
                
                
                #Track participation for KAST calculation
                round_participation[player_names[kill["attackerSteamID"]]] = 1
               
                #Track Deaths
                if  player_names[kill["victimSteamID"]] not in scoreboard.keys():
                    scoreboard[player_names[kill["victimSteamID"]]] = PLAYER_DICT.copy()
                scoreboard[player_names[kill["victimSteamID"]]]["d"] = scoreboard[player_names[kill["victimSteamID"]]].get("d",0)+1
                round_deaths.append(player_names[kill["victimSteamID"]])
                
                #Track Seconds at kill/death
                scoreboard[player_names[kill["victimSteamID"]]]["sd"] = scoreboard[player_names[kill["victimSteamID"]]].get("sd",0)+kill['seconds']
                scoreboard[player_names[kill["attackerSteamID"]]]["sk"] = scoreboard[player_names[kill["attackerSteamID"]]].get("sk",0)+kill['seconds']
                if kill["weapon"] == 'AWP':
                    #Track AWP Kills
                    scoreboard[player_names[kill["attackerSteamID"]]]["awp_kills"] = scoreboard[player_names[kill["attackerSteamID"]]].get("awp_kills",0)+1
                    scoreboard[player_names[kill["victimSteamID"]]]["awp_deaths"] = scoreboard[player_names[kill["victimSteamID"]]].get("awp_deaths",0)+1
                #Track Look anglem for viewX, lets give them a 45 degree "aim" cone
                #attackerViewX
                #victimViewX
                #TODO
                #We want a basic measure of: was the victim looking at the killer, if not, xhair placement can be improved
                anglediff = abs((((kill["attackerViewX"]) - kill["victimViewX"])%360)-180)
                if anglediff > flank_angle:
                    print(kill["victimName"] + " wasn't looking at " + kill["attackerName"])
                    #FLK, Flank kills
                    scoreboard[player_names[kill["attackerSteamID"]]]["flk"] = scoreboard[player_names[kill["attackerSteamID"]]].get("flk",0)+1
                    #FLD, Flank Deaths
                    scoreboard[player_names[kill["victimSteamID"]]]["fld"] = scoreboard[player_names[kill["victimSteamID"]]].get("fld",0)+1
                # AFK, Angle From Killer
                # a *VERY* rough measure of crosshair placement, a statistic that is hard to quantify
                scoreboard[player_names[kill["victimSteamID"]]]["afk"] = scoreboard[player_names[kill["victimSteamID"]]].get("afk",0)+anglediff
                
                #Track Average Distance of death
                scoreboard[player_names[kill["victimSteamID"]]]["ddist"] = scoreboard[player_names[kill["victimSteamID"]]].get("ddist",0)+kill["distance"]
                
                #track advantage at kill time
                scoreboard[player_names[kill["attackerSteamID"]]]["adv"] = scoreboard[player_names[kill["attackerSteamID"]]].get("adv",0)+csgo_stat_functions.calculateKillAdvantage(kill, match_round)
                
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
        #Average Kill Distance
        scoreboard[player]["akd"] = float(scoreboard[player]["dist"]) / float(scoreboard[player]["k"])
        #Average kill time
        scoreboard[player]["skk"] = float(scoreboard[player]["sk"]) / float(scoreboard[player]["k"])
        #Average same econ kills
        scoreboard[player]["asek"] = float(scoreboard[player]["sek"]) / float(scoreboard[player]["k"])
        #Average Eco kills, basically % of kills that were eco kills
        scoreboard[player]["aecok"] = float(scoreboard[player]["ecok"]) / float(scoreboard[player]["k"])
        #% of all kills that were made with a pistol
        scoreboard[player]["pkr"] = float(scoreboard[player]["pk"]) / float(scoreboard[player]["k"])
        # "Ecofragger" Metric
        scoreboard[player]["ecfr"] = float(scoreboard[player]["aecok"]) / float(scoreboard[player]["asek"])
        # Average kills when down on econ
        scoreboard[player]["adek"] = float(scoreboard[player]["dek"]) / float(scoreboard[player]["k"])
    except ZeroDivisionError:
        print(player + " never got a kill, sad")
        scoreboard[player]["hsp"] = 0
    # Classic Kill/Death Ratio
    try:
        scoreboard[player]["kd"] = float(scoreboard[player]["k"]) / float(scoreboard[player]["d"])
        #Average Distance of death
        scoreboard[player]["ddd"] = float(scoreboard[player]["ddist"]) / float(scoreboard[player]["d"])
        #Average death time
        scoreboard[player]["sdd"] = float(scoreboard[player]["sd"]) / float(scoreboard[player]["d"])
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
    
    try:
        #Awp deaths/round
        scoreboard[player]["awdr"] = float(scoreboard[player]["awp_deaths"]) / float(scoreboard[player]["rd"])
        #Awp deaths/death
        scoreboard[player]["awdd"] = float(scoreboard[player]["awp_deaths"]) / float(scoreboard[player]["d"])
        #Awp Duel winrate
        scoreboard[player]["aduel"] = float(scoreboard[player]["awp_kills"]) / float(scoreboard[player]["awp_deaths"])
    except ZeroDivisionError:
        print(player + " never died to an awper, good for them")
        scoreboard[player]["awdr"] = 0
    # Average Advantage per kill
    scoreboard[player]["avadv"] = float(scoreboard[player]["adv"]) / float(scoreboard[player]["k"])
    # avg entry success per t round 
    scoreboard[player]["espr"] = float(scoreboard[player]["ek"]) / float(scoreboard[player]["trd"])
    # flank deaths / round
    scoreboard[player]["fldr"] = float(scoreboard[player]["fld"]) / float(scoreboard[player]["rd"])
    # flank kills / round
    scoreboard[player]["flkr"] = float(scoreboard[player]["flk"]) / float(scoreboard[player]["rd"])
    # average angle from killer
    scoreboard[player]["aafk"] = float(scoreboard[player]["afk"]) / float(scoreboard[player]["d"])
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
        # flash efficiency, seconds blinded / flash thrown
        scoreboard[player]["fr"] = scoreboard[player]["ft"] / scoreboard[player]["rd"]
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

#KMeans calculation

adj_df_score = pd.DataFrame()

adj_df_score.insert(0,"ear",df_score["ear"]) #X

adj_df_score.insert(1,"akp",df_score["akp"]) #Y

adj_df_score.insert(2,"scr",df_score["scr"]) #Z

sc_arr = adj_df_score.values
print(sc_arr)
km = KMeans(n_clusters=4)
km.fit(sc_arr)
labels = km.labels_

df_score.insert(42,"class",labels)

fig = px.scatter_3d(df_score, x = "ear",y="akp",z="scr",color = "class",size="kast",text = df_score.index, symbol = 'team')

fig.show()
fig.write_html("class_kmeans.html")

fig = px.scatter_3d(df_score, x = "adr",y="hsp",z="kd",color = "class",size="kast",text = df_score.index, symbol = 'team')

fig.show()
fig.write_html("performance.html")

fig = px.scatter_3d(df_score, x = "espr",y="ear",z="scr",color = "class",size="kast",text = df_score.index, symbol = 'team')

fig.show()
fig.write_html("entry.html")

fig = px.scatter_3d(df_score, x = "fr",y="far",z="sff",color = "class",size="kast",text = df_score.index, symbol = 'team')

fig.show()
fig.write_html("flash_efficiency.html")

fig = px.scatter(df_score, x = "dpr",y="akd",color = "class",size="kast",text = df_score.index, trendline="ols")

fig.show()
fig.write_html("distance_vs_dpr.html")

fig = px.scatter(df_score, x = "akp",y="akd",color = "class",size="kast",text = df_score.index, trendline="ols")

fig.show()
fig.write_html("awp_vs_distance.html")

fig = px.scatter(df_score, x = "aafk",y="dpr",color = "class",size="kast",text = df_score.index, trendline="ols")

fig.show()
fig.write_html("xhair_placement_vs_deaths.html")

fig = px.scatter(df_score, x = "akp",y="pkr",color = "class",size="kast",text = df_score.index, trendline="ols")

fig.show()
fig.write_html("awp_vs_pistol_kills.html")