# CSGO Stats Analysis
This is a WIP Python project for analyzing individual CSGO players when provided a demo ( or list of demos), with 70+ statistics on each individual, such as:

Angle From Killer: A measure of crosshair placement

Average Advantage Per Kill: A statistic that measures how "useful" a player is, and their impact on the game, advantage is measured based on how many players were alive on each team at the time of the kill, a lower dumber is better, since it denotes that this player can be counted on to even the odds when their team is down players, in a disadvantaged position. 

Average Advantage Per Kill Also helps show the differences between players at each skill rating.

Player Class: A (WIP) basic statistic to group players into roles based on how often they entry, awp, and throw utility/otherwise assist their team. When creating a players class, we avoid performance based metrics as much as possible, as we would like a BAD entry to be in the same class as a GOOD entry, or vice versa.

A handful of economy based statistics, such as Eco Kill%, and Even Econ Kill%: These show how often a player pads their stats with eco kills, or how well they perform in an equal economic situation.

*Economic statistics can be a bit misleading, as a player might have a high % of eco kills, which some might interperet as "Bad" but it simply might mean that the match was very one sided, and that players team had a advantage the whole time.

