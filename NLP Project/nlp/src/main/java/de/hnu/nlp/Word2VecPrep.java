package de.hnu.nlp;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Stream;
import java.util.Arrays;
import java.util.HashMap;


public class Word2VecPrep {
    private String[] words = null;
    private String[] vocabulary = null;

    public Word2VecPrep(String words[]) {
        this.words = words;
    }


    public String[] getVocabulary() {
        HashMap<String,String> vocabulary = new HashMap<String,String>();
        for (String w: words) {
            if (!vocabulary.containsKey(w)) {
                vocabulary.put(w,w);
            }
        }
        Set<String> v = vocabulary.keySet();
        String[] res = new String[v.size()];
        int i = 0;
        for (String s : v) {
            res[i] = s;
            i++;
        }
        this.vocabulary = res;
        return res;
    }

    public void printVector(String word, PrintWriter out) {
        boolean first = true;
        for (int i = 0; i < this.vocabulary.length; i++) {
            if (!first) {
                out.print(",");
            }
            if (word.equals(vocabulary[i])) {
                out.print("1.0");
            } else {
                out.print("0.0");
            }
            first = false;
        }
        out.println();
    }

    public void generateData(String file) {
        try {
            FileWriter fileWriter = new FileWriter(file);
            PrintWriter out = new PrintWriter(fileWriter);

            for (int i = 2; i < words.length-2; i++) {
                System.out.println(words[i-2]);
                printVector(words[i-2],out);
                System.out.println(words[i-1]);
                printVector(words[i-1],out);
                System.out.println(words[i]);
                printVector(words[i],out);
                System.out.println(words[i+1]);
                printVector(words[i+1],out);
                System.out.println(words[i+2]);
                printVector(words[i+2],out);
                System.out.println();
            }

            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        try {
            File file = new File("cats.txt");
            Scanner sc = new Scanner(file);
            String words[] = new String[0];

            while (sc.hasNextLine()) {
                String line = sc.nextLine();
                line = line.replace(".", " eos").toLowerCase();
                String w[] = line.split(" ");
                System.out.println(line);    
                words = Stream.concat(Arrays.stream(words), Arrays.stream(w))
                      .toArray(String[]::new);
            }

            for (String s : words) {
                System.err.println(s);
            }

            Word2VecPrep prep = new Word2VecPrep(words);
            String[] voc = prep.getVocabulary();
            System.out.println(words.length+" "+voc.length);
            prep.generateData("training.csv");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
