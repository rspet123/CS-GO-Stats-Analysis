# CSGO Stats Analysis
This is a WIP Python project for analyzing individual CSGO players when provided a demo ( or folder of demos), with 70+ statistics on each individual, such as:

Angle From Killer: A measure of crosshair placement, roughly measures how far away from their killer a victim was looking. Shows if they're looking at the right place at the right time. This metric also helps show the differences between players at each skill rating, as players get better their AAFK goes down, and the correlation between Deaths/Round and AAPK goes up.

Average Advantage Per Kill: A statistic that measures how "useful" a player is, and their impact on the game, advantage is measured based on how many players were alive on each team at the time of the kill, a lower dumber is better, since it denotes that this player can be counted on to even the odds when their team is down players, in a disadvantaged position. 

Player Class: A (WIP) basic statistic to group players into roles based on how often they entry, awp, and throw utility/otherwise assist their team. When creating a players class, we avoid performance based metrics as much as possible, as we would like a BAD entry to be in the same class as a GOOD entry, or vice versa.
![image](https://user-images.githubusercontent.com/15098644/153897602-0862f68f-11ba-4531-87d8-9d34a6fa8307.png)

A handful of economy based statistics, such as Eco Kill%, and Even Econ Kill%: These show how often a player pads their stats with eco kills, or how well they perform in an equal economic situation.

*Economic statistics can be a bit misleading, as a player might have a high % of eco kills, which some might interperet as "Bad" but it simply might mean that the match was very one sided, and that players team had a advantage the whole time.

