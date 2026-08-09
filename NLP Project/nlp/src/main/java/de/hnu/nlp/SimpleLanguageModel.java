package de.hnu.nlp;

import java.io.File;
import java.util.Scanner;
import java.util.stream.Stream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.ThreadLocalRandom;


public class SimpleLanguageModel {
    private static final int N = 2; // Size of N-Gram
    private HashMap<String,HashMap<String,Integer>> nextWords 
                                            = new HashMap<String,HashMap<String,Integer>>();


    public SimpleLanguageModel(String words[]) {
        for (int i = 0; i < words.length-N; i++) {
            String ngram = words[i];
            for (int j = 1; j < N; j++) {
                ngram = ngram + " " + words[i+j];
            }
            HashMap<String,Integer> next = null;

            if (nextWords.containsKey(ngram)) {
                next = nextWords.get(ngram);
                if (next.containsKey(words[i+N])) {
                    next.put(words[i+N],next.get(words[i+N])+1);
                } else {
                    next.put(words[i+N],1);
                }
            } else {
                next = new HashMap<String,Integer>();
                next.put(words[i+N],1);
                nextWords.put(ngram,next);
            }
        }

        for (String n : nextWords.keySet()) {
            System.out.println(n);

            for (String w : nextWords.get(n).keySet()) {
                System.out.print(w+":"+nextWords.get(n).get(w)+", ");
            }

            System.out.println();
            System.out.println();
        }
    }


    public void generate(String w[], int idx) {
        String ngram = w[0];
        for (int j = 1; j < N; j++) {
            ngram = ngram + " " + w[j];
        }

        if (idx == 0) {
            System.out.print(ngram);
        } else {
            System.out.print(" "+w[N-1]);
        }

        if (nextWords.containsKey(ngram)) {
            HashMap<String,Integer> next = nextWords.get(ngram);

            int cnt = 0;
            for (int c : next.values()) {
                cnt = cnt + c;
            }

            int randomNum = ThreadLocalRandom.current().nextInt(0, cnt);

            cnt = 0;
            for (String nw : next.keySet()) {
                if (randomNum >= cnt && randomNum < cnt+next.get(nw)) {
                    if (idx < 20) {
                        for (int i = 0; i < N-1; i++) {
                            w[i] = w[i+1];
                        }
                        w[N-1] = nw;
                        generate(w,idx+1);
                    }
                    break;
                }
                cnt = cnt + next.get(nw);
            }
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

            HashMap<String,Integer> voc = new HashMap<String, Integer>();

            for (String s : words) {
                System.err.println(s);
                if (voc.containsKey(s)) {
                    voc.put(s,voc.get(s)+1);
                } else {
                    voc.put(s,1);
                }
            }

            System.out.println(voc.size());

            SimpleLanguageModel m = new SimpleLanguageModel(words);
            m.generate(new String[]{ "a", "white" }, 0); // called generate sentences 3 times so that we can get different outputs every time.
            System.out.println();
            m.generate(new String[]{ "a", "white" }, 0); //it creates 3 sentences from the text corpus
            System.out.println();
            m.generate(new String[]{ "a", "white" }, 0);
            System.out.println();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
