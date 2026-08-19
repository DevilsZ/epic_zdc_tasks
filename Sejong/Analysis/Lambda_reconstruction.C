#include <TMath.h>
#include <TVirtualFitter.h>
#include <TGraph2D.h>
#include <TCanvas.h>
#include <TRandom.h>
#include <iostream>
#include <vector>
#include <Math/Vector3D.h>
#include <TError.h>
#include <fstream>

#include "calculate_CM.h"
#include "Si_determineLayer.h"
#include "Hcal_determineLayer.h"
#include "ImagingTopoCluster.h"
#include "ImagingTopoCluster.cc"
#include "PCAtrack.h"
#include "PCAtrack.cc"

using namespace ROOT::Math;

double Hcal_SamplingFraction = 0.02057;
double Si_SamplingFraction = 0.01078;

double crossingangle = -0.025;
double recrossingangle = 0.025;

double Lambda_true_mass = 1.1156;
double Pi0_true_mass = 0.1349768;
double Neutron_true_mass = 0.939565;

double Neutron_p0 = 20.488518;
double Neutron_p1 = 1.190242;

double Gamma_p0 = 0.309672;
double Gamma_p1 = 0.918123;

double Lambda_true_E;
double Neutron_true_E;
double Gamma1_true_E;
double Gamma2_true_E;


void Lambda_reconstruction()
{

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //Open simulation file

    TChain *tin = new TChain("events");  // ← TTree 이름
    ifstream fin("Simulation_filelist.txt");
    string filename;
    while (fin >> filename) {
        TFile *f = TFile::Open(filename.c_str());
        TTree *DATA_Tree = (TTree*)f->Get("events");
        std::cout << "Add file : " << filename << " Entries = " << DATA_Tree->GetEntries() << std::endl;
    
        tin->Add(filename.c_str());
    
        f->Close();

    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //Set ttree valuable 

    TTreeReader tr(tin);

    TTreeReaderArray<float> Si_hits_E(tr, "ZDC_WSi_Hits.energy");
    TTreeReaderArray<Float_t> Si_hits_x(tr, "ZDC_WSi_Hits.position.x");
    TTreeReaderArray<Float_t> Si_hits_y(tr, "ZDC_WSi_Hits.position.y");
    TTreeReaderArray<Float_t> Si_hits_z(tr, "ZDC_WSi_Hits.position.z");

    TTreeReaderArray<float> Hcal_hits_E(tr, "HcalFarForwardZDCHits.energy");
    TTreeReaderArray<Float_t> Hcal_hits_x(tr, "HcalFarForwardZDCHits.position.x");
    TTreeReaderArray<Float_t> Hcal_hits_y(tr, "HcalFarForwardZDCHits.position.y");
    TTreeReaderArray<Float_t> Hcal_hits_z(tr, "HcalFarForwardZDCHits.position.z");

    TTreeReaderArray<Int_t> PDG(tr,"MCParticles.PDG");
    TTreeReaderArray<Double_t> MCparticle_px(tr,"MCParticles.momentum.x");
    TTreeReaderArray<Double_t> MCparticle_py(tr,"MCParticles.momentum.y");
    TTreeReaderArray<Double_t> MCparticle_pz(tr,"MCParticles.momentum.z");
    TTreeReaderArray<Double_t> MCparticle_mass(tr,"MCParticles.mass");
    TTreeReaderArray<Double_t> vertex_x(tr,"MCParticles.vertex.x");
    TTreeReaderArray<Double_t> vertex_y(tr,"MCParticles.vertex.y");
    TTreeReaderArray<Double_t> vertex_z(tr,"MCParticles.vertex.z");

    TTreeReaderArray<double> MCparticle_endx(tr,"MCParticles.endpoint.x");
    TTreeReaderArray<double> MCparticle_endy(tr,"MCParticles.endpoint.y");
    TTreeReaderArray<double> MCparticle_endz(tr,"MCParticles.endpoint.z");

    TTreeReaderArray<Int_t> MCparticle_generatorStatus(tr,"MCParticles.generatorStatus");
    TTreeReaderArray<Int_t> MCparticle_simulatorStatus(tr,"MCParticles.simulatorStatus");

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Define histogram

    TH1D* h_N_Event = new TH1D("h_N_Event","h_N_Event", 10, 0.5, 10.5);
    TH1D* h_N_Lambda = new TH1D("h_N_Lambda","h_N_Lambda", 10, 0.5, 10.5);
    TH1D* h_N_Lambda2 = new TH1D("h_N_Lambda2","h_N_Lambda2", 10, 0.5, 10.5);
    TH1D* h_N_Cluster = new TH1D("h_N_Cluster","h_N_Cluster", 10, 0.5, 10.5);
    TH1D* h_N_new_Cluster = new TH1D("h_N_new_Cluster","h_N_new_Cluster", 10, 0.5, 10.5);
    TH1D* h_invariant_mass_Pi0 = new TH1D("h_invariant_mass_Pi0", "h_invariant_mass_Pi0", 1000, 0.1, 0.2);
    TH1D* h_Lambda_delta_mass = new TH1D("h_Lambda_delta_mass", "h_Lambda_delta_mass", 2000, 0, 2);
    
    TH1D* h_invariant_mass_Lambda_total = new TH1D("h_invariant_mass_Lambda_total", "h_invariant_mass_Lambda_total", 400, 0, 2);
    TH1D* h_invariant_mass_Lambda = new TH1D("h_invariant_mass_Lambda", "h_invariant_mass_Lambda", 400, 0, 2);

    TH1D* h_Neutron_Solidangle = new TH1D("h_Neutron_Solidangle","h_Neutron_Solidangle", 1000, -0.1, 0.1);
    TH1D* h_Gamma_Solidangle = new TH1D("h_Gamma_Solidangle","h_Gamma_Solidangle", 1000, -0.1, 0.1);      
    TH1D* h_Dist_between_Gamma = new TH1D("h_Dist_between_Gamma","h_Dist_between_Gamma", 2000,100,300);

    TH1D* h_Lambda_delta_pT = new TH1D("h_Lambda_delta_pT", "h_Lambda_delta_pT", 2000, -20, 20);
    TH1D* h_Lambda_delta_pz = new TH1D("h_Lambda_delta_pz", "h_Lambda_delta_pz", 1000, -100, 100);
    TH1D* h_Lambda_E_Resolution = new TH1D("h_Lambda_E_Resolution", "h_Lambda_E_Resolution", 1000, -1, 1);
    TH1D* h_Neutron_E_Resolution = new TH1D("h_Neutron_E_Resolution", "h_Neutron_E_Resolution", 1000, -1, 1);
    TH1D* h_Gamma_E_Resolution = new TH1D("h_Gamma_E_Resolution", "h_Gamma_E_Resolution", 1000, -1,1);

    TProfile* h_Ereco_VS_Etrue_photon = new TProfile("h_Ereco_VS_Etrue_photon","h_Ereco_VS_Etrue_photon", 50,0,100);
    TProfile* h_Ereco_VS_Etrue_neutron = new TProfile("h_Ereco_VS_Etrue_neutron","h_Ereco_VS_Etrue_neutron", 100,100,300);

    TH1D* h_NODecayproducts  = new TH1D("h_NODecayproducts" , "Number of Decay products of lambda", 10, 0, 10);
    TH1D* h_true_theta_dist  = new TH1D("h_true_theta_dist" , "h_true_theta_dist", 1000, 0, 1);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Loop Events
    int Signal_Event = 0;
    int N_Event = 0;

    while (tr.Next()) {
        
        h_N_Event->Fill(1);
        N_Event++;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Calculate True Value

        TVector3 Lambda_true_momentum;
        TVector3 Neutron_true_momentum;
        vector<TVector3> Gamma_true_momentum;

        TVector3 Vertex_true_position;

        int N_O_Decayproducts = 0;

        double Lambda_endz;
        double Pi0_endz;
        
        int CountedLambda = 0;
        cout<<"Event: "<<N_Event<<endl;
        for(int i = 0; i < PDG.GetSize(); i++){

            double px = MCparticle_px[i];
            double py = MCparticle_py[i];
            double pz = MCparticle_pz[i];
            double mass = MCparticle_mass[i];

            double E = std::sqrt(px*px + py*py + pz*pz + mass*mass);
            
            if(vertex_z[i] > 35700) continue;

            TLorentzVector p(px, py, pz, E);

            double vx = vertex_x[i];
            double vy = vertex_y[i];
            double vz = vertex_z[i];

            float z = 35800;

            float t = (z-vz)/pz;

            float x = vx + t*px;
            float y = vy + t*py;

            float x_inverse = x*cos(recrossingangle) + z*sin(recrossingangle);
            float y_inverse = y;
            float z_inverse = -x*sin(recrossingangle) + z*cos(recrossingangle);
            
            if(x_inverse > -300 && x_inverse < 300 && y_inverse > -300 && y_inverse < 300){

                if(PDG[i] == 3122){

                    double Lambda_true_theta = 0;

                    Lambda_true_momentum.SetXYZ(MCparticle_px[i],MCparticle_py[i],MCparticle_pz[i]);
                    Lambda_true_theta = TMath::ATan2(sqrt(MCparticle_px[i]*MCparticle_px[i]+MCparticle_py[i]*MCparticle_py[i]),MCparticle_pz[i]);
                    Vertex_true_position.SetXYZ(MCparticle_endx[i],MCparticle_endy[i],MCparticle_endz[i]);                    

                    h_true_theta_dist->Fill(Lambda_true_theta);
                }


                if(PDG[i] == 111 && vz == Vertex_true_position.Z()){

                    Pi0_endz = MCparticle_endz[i];
                }

                if(PDG[i] == 2112 && vz == Vertex_true_position.Z()){

                    Neutron_true_momentum.SetXYZ(MCparticle_px[i],MCparticle_py[i],MCparticle_pz[i]);
        
                    N_O_Decayproducts+=1;
                }

                if(PDG[i] == 22 && vz == Pi0_endz){

                    Gamma_true_momentum.push_back(TVector3(MCparticle_px[i],MCparticle_py[i],MCparticle_pz[i]));
                    
                    N_O_Decayproducts+=1;
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Event selection
        bool ISOK = true;
        if(Si_hits_E.GetSize() == 0) continue;
        if(Hcal_hits_E.GetSize() == 0) continue;
        if (MCparticle_endz.GetSize() == 0) continue;
        // if(Signal_Event > 4) break;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Store hit info

        vector<Hit> hits; // Hit{layer, x pos, y pos, z pos, E, 0 [W-Si] or 1 [Hcal]}
    
        for (unsigned int i=0; i < Si_hits_E.GetSize(); i++) { // Loop W-Si huts

            float Si_hit_E = Si_hits_E[i]/Si_SamplingFraction;
            float Si_hit_x_rotate = Si_hits_x[i]*cos(recrossingangle) + Si_hits_z[i]*sin(recrossingangle);
            float Si_hit_y_rotate = Si_hits_y[i];
            float Si_hit_z_rotate = -Si_hits_x[i]*sin(recrossingangle) + Si_hits_z[i]*cos(recrossingangle);  

            if(Si_hits_E[i] < 7e-5) continue; // hit E cut

            int layer = determineLayer(Si_hit_z_rotate); //layer index 0-19 
            
            hits.push_back(Hit{layer+1, Si_hit_x_rotate, Si_hit_y_rotate, Si_hit_z_rotate, Si_hit_E,0});

        }

        for (unsigned int i=0; i < Hcal_hits_E.GetSize(); i++) { // Loop Hcal htis

            float Hcal_hit_E = Hcal_hits_E[i]/Hcal_SamplingFraction;
            float Hcal_hit_x_rotate = Hcal_hits_x[i]*cos(recrossingangle) + Hcal_hits_z[i]*sin(recrossingangle);
            float Hcal_hit_y_rotate = Hcal_hits_y[i];
            float Hcal_hit_z_rotate = -Hcal_hits_x[i]*sin(recrossingangle) + Hcal_hits_z[i]*cos(recrossingangle);

            if(Hcal_hit_E < 0.02) continue; // hit E cut

            int layer = Hcal_determineLayer(Hcal_hit_z_rotate); // layer index 20-83

            hits.push_back(Hit{layer+1, Hcal_hit_x_rotate, Hcal_hit_y_rotate, Hcal_hit_z_rotate,Hcal_hit_E,1});
            
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Clustering

        //cluster cut
        auto clusters = topo_cluster(
            hits,
            1, //max_layer_diff
            30, //min_cluster_hits
            0.5 //min_cluster_energy
        );

        if(clusters.size() < 3) continue; // Cluster cut

        //Calculate center in each cluster
        std::vector<std::pair<double,double>> cluster_centers;

        for (auto& cl : clusters) {
            double x=0, y=0;
            for (auto& h : cl.hits) {
                x += h.x * h.energy;
                y += h.y * h.energy;
            }
            x /= cl.energy;
            y /= cl.energy;
            cluster_centers.push_back({x,y});
        }

        //Set merged_clsusters(if distance between clusters of cluster is smaller than 10 == same cluster)
        std::vector<Cluster> merged_clsusters;
        std::vector<bool> used(clusters.size(), false);

        for (int i = 0; i < clusters.size(); ++i) {
            if (used[i]) continue;

            Cluster merged = clusters[i];

            for (int j = i + 1; j < clusters.size(); ++j) {
                if (used[j]) continue;

                double dx = cluster_centers[i].first - cluster_centers[j].first;
                double dy = cluster_centers[i].second - cluster_centers[j].second;
                double dist = std::sqrt(dx*dx + dy*dy);

                if (dist < 10) {
                    merged.hits.insert(merged.hits.end(), clusters[j].hits.begin(), clusters[j].hits.end());
                    merged.energy += clusters[j].energy;
                    used[j] = true;
                }
            }
            merged_clsusters.push_back(merged);
        }

        //Arranged in order of highest energy {highest energy = neutron, Gamma, Gamma}
        std::sort(merged_clsusters.begin(), merged_clsusters.end(),[](const Cluster& a, const Cluster& b) {return a.energy > b.energy;});

        //cluster cut
        if(merged_clsusters.size() != 3) continue;
        

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Modify cluster -> n + 2 photon

        // Calculate center of merged cluster
        std::vector<TVector3> merged_cluster_centers;

        for (auto& new_cl : merged_clsusters) {
            double x=0, y=0, z=0;

            for (auto& h : new_cl.hits) {
                x += h.x * h.energy;
                y += h.y * h.energy;
                z += h.z * h.energy;
            }

            x /= new_cl.energy;
            y /= new_cl.energy;
            z /= new_cl.energy;

            double x_rotate = x*cos(crossingangle) + z*sin(crossingangle);
            double y_rotate = y;
            double z_rotate = -x*sin(crossingangle) + z*cos(crossingangle);

            merged_cluster_centers.emplace_back(x_rotate,y_rotate,z_rotate);
        }

        //Identify n cluster in Hcal and 2 photon clusters in W-Si
        double Z_cluster[3];
        for(int i =0; i<3; i++){
            Z_cluster[i] = -merged_cluster_centers[i].X()*sin(recrossingangle) + merged_cluster_centers[i].Z()*cos(recrossingangle);
        }
        bool Identify_photon = true;
        bool Identify_neutron = true;
        if(Z_cluster[0] < 35950 || Z_cluster[0] > 37763) Identify_neutron = false;
        if(Z_cluster[1] < 35810 || Z_cluster[1] > 35950) Identify_photon = false;
        if(Z_cluster[2] < 35810 || Z_cluster[2] > 35950) Identify_photon = false;

        if(Identify_neutron == false || Identify_photon == false) continue;
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Calculate Center of Mass in each layer in each cluster
        
        // Set valuable
        std::vector<float> merged_cl_CMs_E[3];
        std::vector<float> merged_cl_CMs_x[3];
        std::vector<float> merged_cl_CMs_y[3];
        std::vector<float> merged_cl_CMs_z[3];

        for (size_t icl = 0; icl < merged_clsusters.size(); ++icl) {

            auto& merged_cluster = merged_clsusters[icl];

            std::vector<float> merged_cl_hit_E[NUM_LAYERS];
            std::vector<float> merged_cl_hit_x[NUM_LAYERS];
            std::vector<float> merged_cl_hit_y[NUM_LAYERS];
            float merged_cl_hit_z[NUM_LAYERS];

            int icm = 0;

            for (const auto& hit : merged_cluster.hits) {

                int hit_layer = hit.layer;
            
                merged_cl_hit_E[hit_layer - 1].push_back(hit.energy);
                merged_cl_hit_x[hit_layer - 1].push_back(hit.x);
                merged_cl_hit_y[hit_layer - 1].push_back(hit.y);
                merged_cl_hit_z[hit_layer - 1] = hit.z;

            }

            //Calculate CM in each layer
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                if (merged_cl_hit_E[layer].size() > 0)
                {
                    float CM_x, CM_y, CM_z, CM_E;
                    calculate_CM(merged_cl_hit_E[layer], merged_cl_hit_x[layer], merged_cl_hit_y[layer], CM_x, CM_y, CM_E);
                    CM_z = merged_cl_hit_z[layer];

                    double CM_x_rotate = CM_x*cos(crossingangle) + merged_cl_hit_z[layer]*sin(crossingangle);
                    double CM_y_rotate = CM_y;
                    double CM_z_rotate = -CM_x*sin(crossingangle) + merged_cl_hit_z[layer]*cos(crossingangle);
                    
                    merged_cl_CMs_E[icl].push_back(CM_E);
                    merged_cl_CMs_x[icl].push_back(CM_x_rotate);
                    merged_cl_CMs_y[icl].push_back(CM_y_rotate);
                    merged_cl_CMs_z[icl].push_back(CM_z_rotate);
                    
                }
            }
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Calculate Z_vertex

        // e-p Simulation Sclae factor
        double Neutron_recon_E = (merged_clsusters[0].energy+Neutron_p0)/(Neutron_p1);
        double Gamma1_recon_E = (merged_clsusters[1].energy+Gamma_p0)/(Gamma_p1);
        double Gamma2_recon_E = (merged_clsusters[2].energy+Gamma_p0)/(Gamma_p1);

        TVector3 Neutron_recon_center = merged_cluster_centers[0];
        TVector3 Gamma1_recon_center = merged_cluster_centers[1];
        TVector3 Gamma2_recon_center = merged_cluster_centers[2];

        double Dist_between_Gamma = (Gamma1_recon_center - Gamma2_recon_center).Mag();
        double z_dist_Avg_Gamma = (Gamma1_recon_center.z() + Gamma2_recon_center.z())/2;
        h_Dist_between_Gamma->Fill(Dist_between_Gamma);

        double z_vertex_recon = z_dist_Avg_Gamma - 7.4 * Dist_between_Gamma * sqrt(Gamma1_recon_E * Gamma2_recon_E);

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Fitting particles direction

        double x_vertex_recon_rotate = z_vertex_recon * sin(crossingangle);
        double z_vertex_recon_rotate = z_vertex_recon * cos(crossingangle);

        Track3D PCA_recon_track[3];

        int n_cl = 0;   

        for (auto& new_cl : merged_clsusters) {

            std::vector<Eigen::Vector3d> Hits;
            std::vector<double> weights;

            // Input z_vertex_recon
            Hits.push_back(Eigen::Vector3d(x_vertex_recon_rotate,0,z_vertex_recon_rotate));
            weights.push_back(10);

            for (int i = 0; i < merged_cl_CMs_E[n_cl].size(); i++) {
                Hits.push_back(Eigen::Vector3d(merged_cl_CMs_x[n_cl][i], merged_cl_CMs_y[n_cl][i], merged_cl_CMs_z[n_cl][i]));
                weights.push_back(merged_cl_CMs_E[n_cl][i]);
            }

            PCA_recon_track[n_cl] =  FitTrackPCAWeighted(Hits, weights);
            n_cl++;

        }
        
        TVector3 Neutron_recon_dir( PCA_recon_track[0].direction.x(), PCA_recon_track[0].direction.y(), PCA_recon_track[0].direction.z());
        TVector3 Gamma1_recon_dir( PCA_recon_track[1].direction.x(), PCA_recon_track[1].direction.y(), PCA_recon_track[1].direction.z());
        TVector3 Gamma2_recon_dir( PCA_recon_track[2].direction.x(), PCA_recon_track[2].direction.y(), PCA_recon_track[2].direction.z());

        TVector3 Neutron_recon_momentum = (Neutron_recon_dir*sqrt(pow(Neutron_recon_E,2)-pow(Neutron_true_mass,2)));
        TVector3 Gamma1_recon_momentum = (Gamma1_recon_dir*sqrt(pow(Gamma1_recon_E,2)));
        TVector3 Gamma2_recon_momentum = (Gamma2_recon_dir*sqrt(pow(Gamma2_recon_E,2)));
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Recon pi0

        double OpeningAngle_recon = Gamma1_recon_momentum.Angle(Gamma2_recon_momentum);
        double Pi0_recon_mass = sqrt(2*Gamma1_recon_E*Gamma2_recon_E*(1-cos(OpeningAngle_recon)));

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Recon Lambda

        // Reconstruct Lambda
        TVector3 Lambda_recon_momentum = Neutron_recon_momentum + Gamma1_recon_momentum +Gamma2_recon_momentum;
        double Lambda_recon_E = Neutron_recon_E + Gamma1_recon_E + Gamma2_recon_E;
        double Lambda_recon_mass = sqrt(pow(Lambda_recon_E,2) - Lambda_recon_momentum.Mag2());
        

        // Compare recon with true
        if(N_O_Decayproducts == 3 && Gamma_true_momentum.size() == 2){
            Signal_Event++;

            //Calculate true E
            Lambda_true_E = sqrt(Lambda_true_momentum.Mag2() + pow(Lambda_true_mass,2));
            Neutron_true_E = sqrt(pow(Neutron_true_momentum.X(),2)+pow(Neutron_true_momentum.Y(),2)+pow(Neutron_true_momentum.Z(),2)+pow(Neutron_true_mass,2));
            Gamma1_true_E = sqrt(pow(Gamma_true_momentum[0].X(),2)+pow(Gamma_true_momentum[0].Y(),2)+pow(Gamma_true_momentum[0].Z(),2));
            Gamma2_true_E = sqrt(pow(Gamma_true_momentum[1].X(),2)+pow(Gamma_true_momentum[1].Y(),2)+pow(Gamma_true_momentum[1].Z(),2));

            // Mathching photon tracks with true
            TVector3 dir1 = (Gamma1_recon_center-Vertex_true_position).Unit();
            TVector3 dir2 = (Gamma2_recon_center-Vertex_true_position).Unit();
        
            double costA = Gamma_true_momentum[0].Cross(dir1).Mag2() + Gamma_true_momentum[1].Cross(dir2).Mag2();
            double costB = Gamma_true_momentum[0].Cross(dir2).Mag2() + Gamma_true_momentum[1].Cross(dir1).Mag2();
        
            if (costB < costA) std::swap(Gamma_true_momentum[0], Gamma_true_momentum[1]);

            //Calculate angular difference
            double Neutron_Solidangle = Neutron_true_momentum.Angle(Neutron_recon_momentum);            
            double Gamma1_Solidangle = Gamma_true_momentum[0].Angle(Gamma1_recon_momentum);
            double Gamma2_Solidangle = Gamma_true_momentum[1].Angle(Gamma2_recon_momentum);

            //Fill histogram
            h_Ereco_VS_Etrue_photon->Fill(Gamma1_true_E, merged_clsusters[1].energy);
            h_Ereco_VS_Etrue_photon->Fill(Gamma2_true_E, merged_clsusters[2].energy);
            h_Ereco_VS_Etrue_neutron->Fill(Neutron_true_E, merged_clsusters[0].energy);

            h_Neutron_E_Resolution->Fill((Neutron_true_E - Neutron_recon_E)/Neutron_true_E);
            h_Gamma_E_Resolution->Fill((Gamma1_true_E - Gamma1_recon_E)/Gamma1_true_E);
            h_Gamma_E_Resolution->Fill((Gamma2_true_E - Gamma2_recon_E)/Gamma2_true_E);
            
            h_Neutron_Solidangle->Fill(Neutron_Solidangle);

            h_Gamma_Solidangle->Fill(Gamma1_Solidangle);
            h_Gamma_Solidangle->Fill(Gamma2_Solidangle);

            h_invariant_mass_Pi0->Fill(Pi0_recon_mass);

            h_Lambda_E_Resolution->Fill((Lambda_true_E - Lambda_recon_E)/Lambda_true_E);
            h_Lambda_delta_pz->Fill(Lambda_true_momentum.z() - Lambda_recon_momentum.z());
            h_Lambda_delta_pT->Fill(sqrt(pow(Lambda_true_momentum.x(),2)+pow(Lambda_true_momentum.y(),2))-sqrt(pow(Lambda_recon_momentum.x(),2)+pow(Lambda_recon_momentum.y(),2)));            

            h_invariant_mass_Lambda->Fill(Lambda_recon_mass);
        }

        if(Lambda_recon_mass != 0)  h_invariant_mass_Lambda_total->Fill(Lambda_recon_mass);
        
    }

    TString outfile = "./result/Lambda_bg_q21-10.root";
    TFile *outFile = new TFile(outfile, "RECREATE");

    h_N_Event->Write();
    h_N_Lambda->Write();
    h_N_Lambda2->Write();
    h_N_Cluster->Write();
    h_N_new_Cluster->Write();
    h_Neutron_Solidangle->Write();
    h_Gamma_Solidangle->Write();

    
    h_Dist_between_Gamma->Write();
    h_Gamma_E_Resolution->Write();

    h_Neutron_E_Resolution->Write();
    h_Lambda_delta_pT->Write();
    h_Lambda_delta_pz->Write();
    h_Lambda_E_Resolution->Write();

    h_invariant_mass_Pi0->Write();
    h_Lambda_delta_mass->Write();
    h_invariant_mass_Lambda->Write();
    h_invariant_mass_Lambda_total->Write();

    h_Ereco_VS_Etrue_photon->Write();
    h_Ereco_VS_Etrue_neutron->Write();

    h_NODecayproducts->Write();

    outFile->Close();
    
}
