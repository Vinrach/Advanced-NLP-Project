package de.hnu.nlp;

import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Scanner;
import java.io.StringReader;

import com.ugos.jiprolog.engine.JIPDebugger;
import com.ugos.jiprolog.engine.JIPEngine;
import com.ugos.jiprolog.engine.JIPErrorEvent;
import com.ugos.jiprolog.engine.JIPEvent;
import com.ugos.jiprolog.engine.JIPEventListener;
import com.ugos.jiprolog.engine.JIPTerm;

import jep.Interpreter;
import jep.SharedInterpreter;


public class PrologRunner implements JIPEventListener {

    @Override
    public void closeNotified(JIPEvent e) {
        synchronized(e.getSource()) {
            e.getSource().notify();
        }
        System.exit(0);
    }

    @Override
    public void endNotified(JIPEvent e) {
        synchronized(e.getSource()) {
            int nQueryHandle = e.getQueryHandle();
            JIPEngine jip = e.getSource();            
            jip.closeQuery(nQueryHandle);
        }
    }

    @Override
    public void errorNotified(JIPErrorEvent e) {
        synchronized(e.getSource()) {
            int nQueryHandle = e.getQueryHandle();
            JIPEngine jip = e.getSource();            
            jip.closeQuery(nQueryHandle);
        }
    }

    @Override
    public void moreNotified(JIPEvent arg0) {

    }

    @Override
    public void openNotified(JIPEvent arg0) {
    }

    @Override
    public void solutionNotified(JIPEvent e) {
        synchronized(e.getSource()) {
            int nQueryHandle = e.getQueryHandle();            
            System.out.println(e.getTerm());

            JIPEngine jip = e.getSource();            
            jip.nextSolution(nQueryHandle);
        }
    }

    @Override
    public void termNotified(JIPEvent arg0) {
    }

    public static void main(String[] args) {
        String facts = ""; 
        Interpreter python = null;

        try {
            Scanner scanner = new Scanner(Paths.get("teststanza.py"), StandardCharsets.UTF_8.name()); // scans the python script from pah teststanza
            String prog = scanner.useDelimiter("\\A").next();
            scanner.close();

            python = new SharedInterpreter();
            python.exec(prog);
            facts = python.getValue("f", String.class);

            System.out.println("From Java code:");
            System.out.println(facts);
        } catch (Exception e) {
            e.printStackTrace();
        }

        JIPDebugger.debug = true;

        // New instance of prolog engine
        JIPEngine jip = new JIPEngine();                
        System.out.println(jip.getVersion());
        
        // add listeners
        PrologRunner runner = new PrologRunner();
        jip.addEventListener(runner);
                
        JIPTerm query = null;
                            
        try {
            //jip.consultFile("kermit.pl");
            jip.consultStream(new StringReader(facts), "facts");

            synchronized(jip) {
                query = jip.getTermParser().parseTerm("?- hadtobring(X,Y).");
                jip.openQuery(query);
            }

            synchronized(runner) {
                runner.wait();
            }

            python.close();
        } catch(Exception e) {
            e.printStackTrace();
        }     
    }
}
