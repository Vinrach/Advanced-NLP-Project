giveRideToVMAs(kermit, ladyGaga).
doesntRecognize(ladyGaga, security).
leftCredentials(ladyGaga).
jealous(missPiggy,kermit).
jealous(ladyGaga,kermit).
needs(missPiggy,coat).
needs(missPiggy,hat).

endAs(X,Y) :- jealous(Z,X),needs(Z,Y).