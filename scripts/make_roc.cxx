#include <TMVA/ROCCalc.h>
#include <stdio.h>
//#include "mva_read/reader.h"

void  get_roc(TFile * input,TFile* out,string tmp,string flag,bool reverse=false)
{

  
  vector<string> njs = {"2","3"};
  vector<string> prts = {"E","O"};

  vector<TH1D> hists;

  for (auto nj : njs){
    for (auto prt : prts){
      char s_name[100];
      char b_name[100];
      sprintf(s_name,tmp.c_str(),"S",nj.c_str(),prt.c_str());
      sprintf(b_name,tmp.c_str(),"B",nj.c_str(),prt.c_str());
      TH1F* sig = (TH1F*)input -> Get(s_name);
      TH1F* bkg = (TH1F*)input -> Get(b_name);
      
      
      if(reverse){
    //    sig = reverse_h(sig);
    //    bkg = reverse_h(bkg);
      }
      
      TMVA::ROCCalc roc_cal(sig,bkg);
      TH1D * roc = roc_cal.GetROC();
      
      string name = "roc_" + flag + nj + "j" + prt;
      roc -> SetTitle(name.c_str());
      roc -> SetName(name.c_str());

      hists.push_back(*roc);
    }
  }
  out -> cd();
  for(auto h : hists){
    h.Write();
  }
}
void make_roc()
{
  //rnn
  
  TFile * f_rnn = TFile::Open("./outputs/rnn_test_combined.root");
  TFile * out_rnn = new TFile("./data/roc_rnn.root","recreate");
  string tmp_rnn_w = "RNN_%s_TEST_W_%sj%s";
  string tmp_rnn_nw = "RNN_%s_TEST_N_%sj%s";
  
  get_roc(f_rnn,out_rnn,tmp_rnn_w,"test_");
  get_roc(f_rnn,out_rnn,tmp_rnn_nw,"test_nw_");
  out_rnn -> Close();
}
