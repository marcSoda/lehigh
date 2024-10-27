package masa20;

import java.util.*;
import java.util.Map.Entry;

// this data structure represents the coordinates of each seen package.
// the packages are represented by a hashmap of hashmaps which map the packageId to a hashmap of
// coordinates which map to the frequency that that coordinate was reported for that id
// this allows us to keep track of how many times a package was reported at a specific coordinate
// the most frequently reported coordinate for a package is the likely location. this significantly
// reduces if not eliminates phantom packages from being accidently navigated to
// employs a custom comparator which ensures that the freqmap is always sorted with the closest package
// at the front of the array.
public class FreqMap {
    HashMap<Integer, HashMap<Coord, Integer>> packages;

    public FreqMap() {
        packages = new HashMap<>();
    }

    // Add package to the map
    public void addElm(int id, Coord pacCoord, int freq) {
        packages.compute(id, (k, v) -> {
            if (v == null) v = new HashMap<>();
            v.compute(pacCoord, (k2, v2) -> (v2 == null) ? freq : v2 + freq);
            return v;
        });
    }

    // removes all coords with freq < thresh. if that was the only coord, removes the package
    public void trim(Map map) {
        List<Integer> packagesToRemove = new ArrayList<>();
        for (Entry<Integer, HashMap<Coord, Integer>> packageEntry : packages.entrySet()) {
            HashMap<Coord, Integer> coordFreqMap = packageEntry.getValue();
            List<Coord> coordsToRemove = new ArrayList<>();
            for (Entry<Coord, Integer> coordEntry : coordFreqMap.entrySet()) {
                if (coordEntry.getValue() < Brain.thresh)
                    coordsToRemove.add(coordEntry.getKey());
            }
            for (Coord cr : coordsToRemove) {
                coordFreqMap.remove(cr);
                map.map[cr.x][cr.y].type = Map.CellType.EMPTY;
            }
            if (coordFreqMap.isEmpty())
                packagesToRemove.add(packageEntry.getKey());
        }
        for (Integer packageToRemove : packagesToRemove)
            packages.remove(packageToRemove);
    }

    // get a random package
    public Coord getRandomPackageCoord() {
        List<Coord> coords = new ArrayList<>();
        for (HashMap<Coord, Integer> map : packages.values()) {
            for (Coord coord : map.keySet()) {
                if (map.get(coord) > Brain.thresh) {
                    coords.add(coord);
                }
            }
        }
        if (coords.isEmpty()) {
            return null;
        }
        int randIndex = (int) (Math.random() * coords.size());
        return coords.get(randIndex);
    }

    // get all package locations. used for sending package location message
    public String getMostFrequentCoords() {
        StringBuilder sb = new StringBuilder();
        for (Entry<Integer, HashMap<Coord, Integer>> packageEntry : packages.entrySet()) {
            Coord mostFrequentCoord = getMostFrequentCoord(packageEntry.getValue());
            if (mostFrequentCoord != null) {
                sb.append(packageEntry.getKey()).append("|")
                    .append(mostFrequentCoord.toString().replaceAll("\\[|\\]", ""))
                    .append("\n");
            }
        }
        return sb.toString();
    }

    // delete package by id
    public void del(int id) {
        packages.remove(id);
    }

    // delete package by coord
    public void del(Coord coord) {
        packages.entrySet().removeIf(entry -> entry.getValue().containsKey(coord));
    }

    // get the most frequent coord by package id (helper for next func)
    Coord getMostFrequentCoord(int id) {
        HashMap<Coord, Integer> map;
        map = this.packages.get(id);
        return getMostFrequentCoord(map);
    }

    // get the most frequent coordinate for a package
    private Coord getMostFrequentCoord(HashMap<Coord, Integer> map) {
        Coord maxCoord = null;
        int maxFreq = 0;
        for (Entry<Coord, Integer> entry : map.entrySet()) {
            if (entry.getValue() > maxFreq) {
                maxFreq = entry.getValue();
                maxCoord = entry.getKey();
            }
        }
        return maxCoord;
    }

    // given a coord, return its max frequency
    public int getMaxCountForCoord(Coord coord) {
        int maxCount = 0;
        for (HashMap<Coord, Integer> map : packages.values()) {
            Integer count = map.get(coord);
            if (count != null && count > maxCount) {
                maxCount = count;
            }
        }
        return maxCount;
    }

    // get the coordinate of the closest package.
    // if the frequency is below bran.thresh, it will delete it
    public Coord getClosestPackageCoord(Coord me, Map map) {
        List<Entry<Integer, HashMap<Coord, Integer>>> list = new ArrayList<>(packages.entrySet());
        if (list.isEmpty()) return null;
        Collections.sort(list, new FreqComp(me));
        for (Entry<Integer, HashMap<Coord, Integer>> packageEntry : list) {
            Coord mostFrequentCoord = getMostFrequentCoord(packageEntry.getValue());
            int frequency = packageEntry.getValue().get(mostFrequentCoord);
            if (frequency > Brain.thresh) {
                return mostFrequentCoord;
            } else {
                packages.remove(packageEntry.getKey());
                map.map[mostFrequentCoord.x][mostFrequentCoord.y].type = Map.CellType.EMPTY;
            }
        }
        return null;
    }

    // comparator to sort packages by frequency of closest coordinate to me
    private class FreqComp implements Comparator<Entry<Integer, HashMap<Coord, Integer>>> {
        Coord me;
        FreqComp(Coord c) {
            this.me = c;
        }
        public int compare(Entry<Integer, HashMap<Coord, Integer>> e1, Entry<Integer, HashMap<Coord, Integer>> e2) {
            Coord c1 = getMostFrequentCoord(e1.getValue());
            Coord c2 = getMostFrequentCoord(e2.getValue());
            return Integer.compare(c1.dist(me), c2.dist(me));
        }
    }
}
