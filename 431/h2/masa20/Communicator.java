package masa20;

import pacworld.*;

// handles communication
class Communicator {
    Brain brain;
    StringBuilder status = new StringBuilder();

    Communicator(Brain brain) {
        this.brain = brain;
    }

    // dispatch message to apropriate parser
    void parse(String m) {
        if (m.charAt(0) == 'd') parseDel(m);
        else parsePacks(m);
    }

    // broadcast package locations
    Say packs() {
        brain.map.pacMap.trim(brain.map);
        String stat = brain.map.pacMap.getMostFrequentCoords();
        if (stat.length() == 0) return null;
        return new Say(stat);
    }

    // parse and apply received package locations
    void parsePacks(String m) {
        String[] lines = m.split("\n");
        for (String line : lines) {
            String[] pair = line.split("\\|");
            int id = Integer.parseInt(pair[0]);
            String[] strC = pair[1].split(",");
            Coord coord = new Coord(Integer.parseInt(strC[0]), Integer.parseInt(strC[1]));
            brain.map.pacMap.addElm(id, coord, Brain.thresh+1);
            brain.map.setCoordType(coord, Map.CellType.PACK);
        }
    }

    // broadcast package delete message
    Say del(Coord c) {
        String del = "d" + c.toString().replaceAll("\\[|\\]", "");
        return new Say(del);
    }

    // parse package delete message
    void parseDel(String m) {
        String[] strC = m.substring(1).split(",");
        Coord coord = new Coord(Integer.parseInt(strC[0]), Integer.parseInt(strC[1]));
        brain.map.erasePackage(coord);
    }
}
