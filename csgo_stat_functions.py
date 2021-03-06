#A Handful of useful functions for calculating stats based on round frames and kill data
from csgo.parser import DemoParser
    
def matchKillToFrame(kill: dict, match_round: dict):
    frame_after_kill = None
    frame_before_kill = None
    try:
        kill_tick = kill['tick']
    except KeyError:
        print("Wrong Dict Type")
        return 0
    for i, frame in enumerate(match_round['frames']):
        if frame['tick'] > kill_tick:
            frame_before_kill = match_round['frames'][i]
            frame_after_kill = frame
            return (frame_before_kill,frame_after_kill)
    print("Kill is not in round?")
    return 0
    
    
def calculateKillAdvantage(kill: dict, match_round: dict, debug = False):
    #frame_after = matchKillToFrame(kill,match_round)[1]
    try:
        frame_before = matchKillToFrame(kill,match_round)[0]
    except TypeError:
        return 0
    
    #friendly_players_after_kill = frame_after[kill['attackerSide'].lower()]['alivePlayers']
    friendly_players_before_kill = frame_before[kill['attackerSide'].lower()]['alivePlayers']
    enemy_players_before_kill = frame_before[kill['victimSide'].lower()]['alivePlayers']
    #enemy_players_after_kill = frame_after[kill['victimSide'].lower()]['alivePlayers']
    
        
    advantage = friendly_players_before_kill-enemy_players_before_kill
    if debug:
        #TODO
        #Seems a bit messed up
        if advantage < 0:
            print(kill['attackerTeam'] + " are down " + str(abs(advantage)))
        elif advantage == 0:
            print("teams are even")
        else:
            print(kill['attackerTeam'] + " is up " + str(abs(advantage)))
    return advantage
    

def thread_process_demo(demo_path: str,demos_dict: dict):
    demos_dict.append(DemoParser(demofile="resources\\" + demo_path, demo_id=demo_path,outpath="JSONLogs", parse_rate=128, parse_frames=True).parse())