void runDDSim() {

    int max_jobs = 40;
    int jobs_per_energy = 40;
    int events_per_job = 250;

    int E[5] = {200, 200, 200, 200, 200,};
    double Theta[5] = {0, 0.001, 0.002, 0.003, 0.004};

    int job_counter = 0;

    for (int e = 0; e < 1; e++) {

        gSystem->Exec(Form("mkdir -p ./Silicon/lambda/%dGeV_%frad", E[e], Theta[e]));

        for (int i = 0; i < jobs_per_energy; i++) {

            if (job_counter > 0 && job_counter % max_jobs == 0)
                gSystem->Exec("wait");

            int seed = 100000 + job_counter;

            gSystem->Exec(Form(
                "ddsim --steeringFile Makesteering.py "
                "--compactFile $DETECTOR_PATH/epic_craterlake_18x275.xml "
                "--outputFile ./Silicon/lambda/%dGeV_%frad/lambda_%dGeV_job_%d.root "
                "-G -N %d "
                "--random.seed %d "
                "--gun.particle lambda "
                "--gun.energy %d*GeV "
                "--gun.position '(0, 0, 0 * mm)' "
                "--gun.thetaMin %f*rad "
                "--gun.thetaMax %f*rad "
                "--gun.phiMin 0.5*pi*rad "
                "--gun.phiMax 0.5*pi*rad "
                "--gun.distribution uniform "
                "--crossingAngleBoost -0.025 &",
                E[e], Theta[e], E[e], i,
                events_per_job,
                seed,
                E[e],
                Theta[e], Theta[e]
            ));

            job_counter++;
        }
    }

    gSystem->Exec("wait");
}
