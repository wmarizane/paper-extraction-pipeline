Article 

pubs.acs.org/Macromolecules 

## Comparison of Critical Adsorption Points of Ring Polymers with Linear Polymers 

Jesse D. Ziebarth, Abigail Anne Gardiner, and Yongmei Wang* 

Department of Chemistry, The University of Memphis, Memphis, Tennessee 38154, United States 

## Youncheol Jeong, Junyoung Ahn, Ye Jin, and Taihyun Chang* 

Division of Advanced Materials Science and Department of Chemistry, Pohang University of Science and Technology (POSTECH), Pohang 37673, Korea 

## *S Supporting Information 

ABSTRACT: The critical adsorption points (CAP) for ring and linear polymers are determined and compared using Monte Carlo simulations and liquid chromatography experiments. The CAP is defined as the coelution point of ring or linear polymers with different molecular weights (MW). Computational studies show that the temperature at the CAP, TCAP, for rings is higher than TCAP for linear polymers regardless of whether the chains are modeled as random walks or self-avoiding walks. The difference in the CAP can be attributed only to the architectural difference. Experimentally, four pairs of linear and ring polystyrenes (PS) of different MW were synthesized and purified. Care was taken to account for 

the difference between the end-groups in linear polymers and the linkage unit in ring polymers. Elution of these polymers using a C18 bonded silica stationary phase and a CH2Cl2/CH3CN mixed eluent were studied. The temperature at the coelution point, TCAP, and the coelution time at the CAP, tE,CAP, were determined for both ring and linear polymers. Experimentally, it was found that TCAP of linear PS is lower than TCAP of cyclic PS and tE,CAP of linear PS is shorter than tE,CAP of ring PS. Therefore, at the CAP of linear polymers, ring polymers elute later in order of increasing MW while, at the CAP of ring polymers, linear polymers elute earlier in order of decreasing MW. This is in excellent agreement with the Monte Carlo computer simulation results. We also found that the functionality effect can interfere in the LCCC separation of ring polymers from their linear precursors. 

## I. INTRODUCTION 

The critical adsorption point (CAP) of a polymer near a solid surface is an interesting phenomenon in the study of polymer adsorption.[1] It is well-known that if the polymer/surface interaction εw is more attractive than the CAP, then the polymer will be adsorbed on the surface; on the other hand, if the polymer/surface interaction εw is less attractive than the CAP, then the polymer will not be adsorbed on the surface and, instead, will be repelled by the surface.[2][−][7] Here the polymer/ surface interaction εw is a reduced interaction parameter that can be written as εw = ΔEw/kBT, where ΔEw represents the energy change associated with the formation of a polymer surface contact, kB is the Boltzmann constant, and T is the temperature. Hence, increasing the temperature will effectively reduce the attractive interaction and lowering the temperature will effectively increase the adsorption provided that the adsorption is caused by a short-range attractive interaction (i.e., ΔEw < 0). The CAP can be expressed either as εw(CAP) or as a temperature, TCAP, and the two are interconvertible. 

While the existence of the CAP is well accepted, providing a precise definition of the CAP can be complex. If one uses a 

random walk model to represent the polymer chain, then the precise value of the CAP, though it depends on the lattice type, is independent of chain length. However, consideration of the excluded volume effect in a real polymer chain makes the adsorption transition chain length dependent.[6] Therefore, the definition of the CAP for chains with excluded volume effect is normally defined at the infinitely long chain limit.[5][,][6] The physical meaning of the CAP can be understood as the point where the monomeric free energy of a chain near the surface becomes equal to the monomeric free energy of a chain in the bulk solution. However, several physical properties of chains, such as their radius of gyration, exhibit a crossover behavior near the CAP that can be utilized to determine the CAP in computer simulations.[8][,][9] 

Experimental determination of the CAP is nearly impossible except in the field of liquid chromatography. In the past two decades, polymer liquid chromatography has developed the socalled liquid chromatography at the critical condition (LCCC) 

Received: August 31, 2016 Revised: October 27, 2016 Published: November 3, 2016 

8780 

© 2016 American Chemical Society 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

Article 

## Macromolecules 

where homopolymers with different molecular weights (MW) coelute at the LCCC.[10][−][15] Ample evidence suggests that this LCCC condition coincides with or is closely related to the CAP.[14][,][16][−][21] Experimentalists typically achieve this LCCC condition by changing the eluent composition and/or the column temperature to achieve a point where polymers with different MW coelute. Comparison of LCCC conditions found for different types of polymers usually will not yield useful information since the LCCC condition depends on the chemical nature of polymer repeating units, chromatography columns, and solvents. However, comparison of LCCC conditions for the same type of polymers but with different architecture, such as linear chains versus rings, would reflect how polymer architecture impacts polymer elution behavior in chromatography. This is the focus of the current study. 

Ring polymers are a special class of polymers that have attracted many studies. Ring polymers have an end-to-end distance of zero, and a knot in a ring cannot be unknotted without bond breaking and re-formation. Because of these topological characteristics, conformational statistics and dynamics of ring polymers differ much from their linear counterparts.[22][−][27] Adsorption of ring polymers was compared against adsorption of linear polymers,[28][−][30] and some studies concluded that ring polymers were more adsorbed than the linear polymers[30] although others suggested otherwise.[28][,][29] Recent attention has turned to ring polymers under confinements.[31][−][35] One interesting phenomenon is the segregation of ring polymers in cylindrical confinement which was shown to be more significant and faster than linear polymer. This could be the mechanism that facilitates the segregation of bacteria’s circular chromosome during cell division.[36] 

The different behavior of ring and linear polymers under confinement encountered in chromatographic separations has also been utilized to prepare ring polymers of high purity. Ring polymers are generally prepared via a ring closure reaction of telechelic linear polymer precursors. However, contamination of byproducts in the target ring polymer is unavoidable due to imperfect ring-closure reactions.[37][−][41] Therefore, postfractionation is necessary in order to obtain pure ring polymers. LCCC separation at the CAP of linear polymers is known to be the best available method for obtaining highly pure ring polymers from the ring closure reaction mixture.[37][,][41][−][46] At the LCCC condition of linear polymer, the ring polymers are typically found to be eluted after the linear polymers. A theoretical prediction of the LCCC separation of ring polymers from linear polymers based on Gaussian chain statistics was compared with experimental results.[44] It was shown that at the LCCC condition of the linear polymer the partition coefficient K for rings is greater than one and increases with the ring MW, in qualitative agreement with experimental results. 

The above discussion leads to an interesting question: will there be a CAP for ring polymers and, if it exists, how is the CAP of ring polymers compared with the linear polymers? Here the CAP is defined as the coelution point of a set of ring or linear polymers with different MW but with the exact same chemical repeating units. This question has not been fully addressed experimentally and theoretically. LCCC retention depends rather significantly on the functionality of polymers, and LCCC has been used widely to separate polymers according to the functionality.[10][,][47][−][49] Ring polymers are often prepared from linear precursors that contain functional end-groups that are used to link the chain ends in the ring-closure reaction. The difference between the end-groups in precursors and the linkage unit in the 

ring polymers should affect their LCCC retention to some extent. Hence, in order to truly compare the retention times of linear and ring polymers due to difference in architecture, not in functionality, one must be careful to take the end-group differences into account. In this study, we prepared a set of linear and ring polystyrenes (PS) of different MW that have a minimal difference in their functionality and investigate the elution behavior of both ring polymers and their linear precursors at their respective CAPs. The experimental results are compared with computational investigations of the CAP of ring and linear polymers. 

## II. METHODS 

II.A. Computer Simulations. Polymer chains are represented by either self-avoiding walks (SAW) or random walks (RW) on a simple cubic lattice. We consider polymer chains partitioning between a slit pore consisting of two solid impenetrable walls and a bulk solution. We used a simulation box consisting of two regions: Region A is tetragonal box of size 50 × 50 × D with two solid walls in the z-direction at z = 1 and z = D. A pairwise polymer segment and wall interaction εw is applied whenever the segment is in region A and is in z = 2 and z = D − 1 layers. The other region B is a simple cubic box of 50 × 50 × 50 with periodic boundary conditions in all directions. The two boxes are connected along the x-direction so that the chains can freely move between region A and region B. Polymer chains are initially placed in region B and are allowed to move and equilibrate between the two regions by subjecting the chains to reptation moves (applied only to linear polymer chains) and local moves. The Metropolis rule is applied in every step. The density profiles of polymer chains along the z-direction in regions A and B are then collected and averaged over the layers in the two regions but not including the interfacial layers (about 10 layers) between regions A and B. This ensures that the density profile collected in region B, ϕB, reflects a bulk solution and the density profile in region A, ϕA, reflects chains confined in the slit. The partition coefficient K is calculated based on K = ⟨ϕ⟩A/⟨ϕ⟩B, and this is directly correlated with the elution time tE in experiments. This twinbox method has been used in previous studies.[50][−][52] 

In the current study, we focused on chain lengths, N, that include 20, 40, 80, and 100. In order to determine the CAP, simulations are performed with discrete εw values, and a histogram reweighting method[53][,][54] is then used to combine runs and recalculate the partition coefficient K as a continuous function of εw. The Appendix discusses how the histogram reweighting method is applied here to calculate K. The histogram reweighting method improved the statistics of the data significantly. Figure S1 in the Supporting Information shows how the calculated K from the histogram reweighting method interpolates the discrete K determined directly from each individual run. All the simulation results presented are K values calculated using the histogram reweighting method and will be shown as lines instead of discrete data points. 

For linear polymer chains, we can also use the biased chain insertion method to determine the chemical potential of the chain μ[0] bulk[and][ μ][0] slit when the chain is inserted either in a simulation box like region A only or in a box like region B, respectively. These give a partition coefficient at infinite dilution K0 = exp[−(μ[0] slit[−][μ][0] bulk[)]/][k] B[T][.][All][our][previous] computational studies on LCCC used the biased chain insertion method.[21][,][55][,][56] The partition coefficient K determined using the twinbox simulation is at finite bulk concentration, ⟨ϕ⟩b, which in this current study is controlled at 0.01 or less. This concentration is dilute enough for short chains such that K obtained from the twinbox simulation agrees well with K0 obtained from biased chain insertion. Figure S1 also compares the K calculated from twinbox simulations with the K0 from biased chain insertion method and demonstrates the agreement between two different approaches. Another set of runs with smaller ⟨ϕ⟩b makes the line for K in Figure S1 slightly steeper, giving even better agreement with K0 values determined via the biased chain insertion method. For ring polymers, the biased chain insertion method 

8781 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

Macromolecules 

Article 

## Scheme 1. Synthetic Scheme of PS Precursor and Ring PS 

Figure 1. Partition coefficient K as a function of |εw| showing the crossover point at the CAP. Left panel is for the linear chain, and right panel is for the ring polymer. Both results are for random walk chains partitioning into a slit of width D = 20. 

can no longer be applied. We determined K through the twinbox simulation only. 

II.B. Experimental Section. Preparation of Ring Polystyrene. The initiator, potassium naphthalenide, was synthesized by the reaction between potassium (Aldrich, 99%) and naphthalene (Aldrich, 99%) in THF at room temperature for 5 h. 1,1-Diphenylethylene (DPE, Hokko Chemicals, 95%) was dried over calcium hydride first and treated with n- BuLi before vacuum distillation. 1-(4-(3-Bromopropyl)phenyl)-1phenylethylene (DPE-Br) was synthesized according to a method in the literature.[57] All polymerizations and reactions were carried out under an Ar atmosphere. The preparation scheme of telechelic polystyrene with a DPE functional group at both ends is shown in Scheme 1. Anionic polymerization of styrene was initiated by potassium naphthalenide in THF at −78 °C. After 2 h of polymerization, THF solution of DPE (0.5 mol/L) was added for end-capping at a molar ratio of [DPE]/[K[+] ] ∼ 3 and was allowed to react for 30 min. Then THF solution of DPE-Br (0.5 mol/L) was added at a molar ratio of [DPE-Br]/[K[+] ] ∼ 3 at −78 °C. The end-capped linear PS precursors were precipitated in an excess 

amount of methanol and then reprecipitated in acetonitrile to remove the residual DPE and DPE-Br. The ring closure reaction of the PS precursors was carried out by slowly introducing potassium naphthalenide (0.071 mol/L) in THF into a very dilute THF solution (0.2−0.5 g/L) of the PS precursor at a molar ratio of [DPE]/[K[+] ] ∼ 1/5 at 25 °C and stirred for 5 h. After quenching with dried methanol, the polymer was precipitated in an excess amount of methanol. The unlinked functional ends on PS precursor after the methanol quench become saturated. Therefore, the linear polymers recovered from the purification of the ring products have saturated ends (labeled as Ls-PS) and differ from the linear PS precursor which has unsaturated reactive end groups (labeled as Lu-PS). 

SEC Characterization. All polymers used in this study were characterized by SEC/light scattering detection. The specific refractive index increment (dn/dc) of PS was measured in THF as 0.185 mL/g. Three styrene gel columns (Agilent Polypore 300 × 7.5 mm, Waters Styragel HR4 300 × 7.8 mm, and Jordi mixed bed 300 × 8.0 mm) were used with THF (Samchun, HPLC grade) as an eluent at a flow rate of 0.7 

8782 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

Macromolecules 

Article 

mL/min. The column temperature was controlled at 40 °C. SEC chromatograms were recorded with a triple detector (Malvern, TDA 300). Polymer samples for the SEC analysis were dissolved in THF at a concentration of ∼1 mg/mL, and the injection volume was 100 μL. BY LCCC Fractionation of Ring PS. For fractionation of ring PS, two C18 bonded silica columns (Nucleosil C18, 250 × 9.8 mm, 100 Å pore, 5 μm and Nucleosil C18, 250 × 9.8 mm, 300 Å pore, 5 μm) and a mixed RT eluent of CH2Cl2/CH3CN (58/42, v/v, Samchun, HPLC grade) at a FD flow rate of 2 mL/min were used. The temperature of the columns was set at 19 °C at which different molar mass linear PS precursors coelute. Sample solutions (∼10 mg/mL) were prepared by dissolving the polymers in a small volume of the eluent, and the injection volume was eee” 500 μL. The chromatograms were recorded by a light scattering detector eee (Wyatt, miniDAWN) and a UV absorption detector (Younglin, > UV7300) operating at a wavelength of 260 nm. LCCC Analysis. Analytical LCCC analysis used a column (Nucleosil C18, 250 × 4.6 mm, 100 Å, 5 μm) smaller than the preparative fractionation. The eluent was CH2Cl2/CH3CN (58/42, v/v) at a flow rate of 0.5 mL/min. The column temperature was varied to achieve CAP ee QI.» condition for the polymers of interest. 

chain is defined by the condition that λ = 0 where the parameter λ is related to the derivative of partition function of the chain near the surface. They showed that at the CAP for linear polymer (λ = 0) the partition coefficient K for rings is greater than one and increases with the ring size. Although they did not define the CAP for the ring polymer, their data clearly suggest that the coelution of ring polymers occurs when λ > 0, implying that ring coelution will occur when linear chains are in SEC mode (nonadsorptive). Figure 3 presents individual K values at either εwRW(CAP, linear) or εwRW(CAP, ring) identified in Figure 1. When the surface interaction is set at εwRW(CAP, linear), K for the ring polymers are greater than 1.0 and K increases with chain length N, behavior typical of IC mode. When the surface interaction is set at εwRW(CAP, ring), we see that K for the linear chains exhibits the typical SEC mode elution. 

When we switch from the RW to the SAW chain model, we observe very similar behavior. Figure 4 presents the plot of K versus |εw| for different chain lengths for both linear and ring polymers modeled as SAW chains. We see that the lines for the ring polymers intersect at |εwSAW(CAP, ring)| = 0.246, smaller than the CAP for the linear polymers, |εwSAW(CAP, linear)| = 0.269. We note that the |εwSAW(CAP, linear)| identified using twinbox simulation data is slightly smaller than the value identified earlier using K0 values obtained via the biased insertion method.[20][,][21] This is partially due to the fact that K determined here is at finite bulk concentration. Figure 5 shows the plot of standard deviation in ln K versus |εw| and shows |εwSAW(CAP)| clearly. We include the data for two different slit widths, D = 20 and D = 30, for both linear and ring polymer chains. The impact of variation in slit width on ring polymers resembles its impact on linear chains. The CAP identified in the two slit widths have little or no shift for either the linear or ring polymers. These computational results clearly show that TCAP(ring) > TCAP(linear) when all the repeating units on ring and linear chains are the same. 

## III. RESULTS AND DISCUSSION 

Computer Simulations. We first present the results obtained for the RW model of chain in a simple cubic lattice. The CAP for RW of linear polymer chain is well-known and is given by |εwRW(CAP, linear)| = 0.182 32 or TCAPRW(linear) = 5.4848 (note: T = 1/|εw|). Earlier we have used the biased insertion method to determine K0 for linear chains and confirmed this CAP value. Figure 1 presents how K obtained for linear chains from twinbox simulations intersect exactly at this known value: |εwRW(CAP, linear)| = 0.182. The ring polymers, however, intersect at |εwRW(CAP, ring)| = 0.171 or TCAPRW(ring) = 5.848, higher than TCAPRW(linear). Also noticeable is that the lines for the ring polymer do not overlap as well as for the linear chain. Figure 2 presents the plot of standard deviation in ln K for 

Preparation and Characterization of Ring PS. Among the various methods to prepare ring polymers, we chose a method that renders minimal difference in functionality (of units other than the repeating units of PS) between linear and ring polymers since functionality can affect HPLC retention behavior of polymers.[47][−][49] We prepared PS precursors of four different MW (16K, 33K, 65K, and 92K) according to the procedure shown in Scheme 1. SEC chromatograms of the PS precursors (dashed line) and the cyclization products (solid line) are displayed in Figure 6. In addition to the desired single rings that elute after the linear precursors due to its smaller size, the ringclosure reaction apparently also yields various polycondensation byproducts that elute before the linear precursors. The weightaverage MW (Mw) of the precursor PS was measured by light scattering detection, and the polydispersity index was determined by the SEC-LS method as Mw/Mn < 1.01. 

**==> picture [260 x 13] intentionally omitted <==**

**----- Start of picture text -----**<br>
g [/] other<br>**----- End of picture text -----**<br>


Figure 2. Replot of data in Figure 1 using the standard deviation in ln K versus |εw| for the linear and ring polymers identifies |εwRW(CAP)| = RW 0.182 for linear chain and |εw (CAP)| = 0.171 for the ring. 

The cyclization products were fractionated by LCCC at the CAP of the linear PS precursors. Figure S2 shows the LCCC chromatogram to fractionate the ring PS from the linear PS for the cyclization product of 92 kg/mol PS. Both ring and linear parts obtained by LCCC fractionation contain not only unimer but also dimer as well as higher condensation products as shown in Figure S3. They were further fractionated by IC to obtain unimers for both ring (Figure S4) and linear PS (Figures S5 and S6). The linear PS unimers fractionated from the cyclization product differ from the linear PS precursors in the endfunctionality. The PS precursors before the cyclization reaction contain 1,1-diphenylethylene (unsaturated-DPE) moiety at both 

these chain length vs surface interaction which gives a minimum RW corresponding to εw(CAP). As can be seen, the |εw (CAP, RW ring)| < |εw (CAP, linear)|, and there is more deviation in K for rings at the CAP than for linear chains. This implies that the CAP for ring polymers occurs at a less attractive surface interaction than for linear polymers or TCAPRW(ring) > TCAPRW(linear). 

These results are consistent with theoretical studies using the Gaussian chain model. Gorbunov and Vakhurshev examined the partitioning of rings from the size exclusion chromatography (SEC) mode to the interaction chromatography (IC) mode.[58] Within the Gaussian chain model, the CAP for the linear polymer 

8783 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

Macromolecules 

Article 

Figure 3. Dependence of the partition coefficient K on the chain length for the ring and linear polymers when segment/surface interaction |εw| is (a) at |εw(CAP, linear)| = 0.182 and (b) at |εw(CAP, ring)| = 0.171. 

Figure 4. Partition coefficient K as a function |εw| for SAW chain model; solid lines are for linear polymers and dashed lines are for ring polymers. Slit width D = 30. 

Figure 6. SEC chromatograms of four PS precursors (dashed line) and corresponding cyclization products (solid line) recorded by a differential refractive index detector. 

account of end group effect in addition to the effect of chain architecture. 

Figure 7 shows the SEC chromatograms of the fractionated ring PS unimers (solid line, Ring-PS) and the precursor unimers 

Figure 5. Plot of standard deviation in ln K for chain length N varied from 20, 40, 80 to 100 as a function of |εw| in slit width D = 20 and D = 30 for linear and ring chains. Chains are modeled as self-avoiding walks; rings are unknotted and self-avoiding. Identified |εw|[SAW] (CAP, linear) = 0.269; |εw|[SAW] (CAP, ring) = 0.246. 

chain ends for the ring-closure reaction while the fractionated linear unimer from the cyclization product has 1,1-diphenylethyl (saturated-DPE) moiety since an excess amount of K- naphthalenide was used in the cyclization reaction. The saturated-DPE is more similar to the linkage unit in the ring PS (Figure S7). We will compare the retention behavior of both linear PSs with saturated end-groups (Ls-PS) and unsaturated end-groups (Lu-PS) against the ring polymers to take into 

Figure 7. SEC chromatograms of Ls-PS unimer (dashed line) and RingPS unimer (solid line) fractionated through two-step LCCC/IC fractionation. 

8784 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

Article 

## Macromolecules 

with saturated DPE end-groups (dashed line, Ls-PS). The ring polymers are eluted later than the linear PS since the rings have smaller size and SEC separates polymers according to the size. The MW of the fractionated samples and precursors with unsaturated DPE end-groups (Lu-PS) were determined by light scattering detection, and the results are summarized in Table 1. The MW of the three PS samples sets are practically identical. 

Table 1. Molecular Characteristics of Linear and Ring PS Used in This Study[a] 

|Lu-PS<br>samples<br>Lu-PS1|Mw<br>(kg/mol)<br>16.4|Ls-PS<br>samples<br>Ls-PS1|Mw<br>(kg/mol)<br>16.4|ring PS<br>samples<br>R-PS1|Mw<br>(kg/mol)<br>16.8|
|---|---|---|---|---|---|
|Lu-PS2|33.1|Ls-PS2|32.8|R-PS2|34.2|
|Lu-PS3|64.6|Ls-PS3|64.0|R-PS3|65.5|
|Lu-PS4|92.2|Ls-PS4|91.8|R-PS4|92.6|



> aMw/Mn values determined by SEC-LS method are less than 1.01. 

Elution Behavior of Ring and Linear PS at the CAP of the Counterpart. Figures 8a, 8b, and 8c show the LCCC chromatograms of the three sets of PS samples (Ls-PS, Lu-PS, and Ring-PS of four different MW: 16K, 33K, 64K, and 92K) at the CAP of Lu-PS, Ls-PS, and Ring-PS, respectively. The three PS samples differing in their chain architecture and end-groups elute in the sequence of Ls-PS, Lu-PS, and Ring-PS near CAP of the samples. TCAP and tE,CAP of Ls-PS, Lu-PS, and Ring-PS are measured as 14.8 °C/7.4 min, 14.95 °C/7.6 min, and 17.3 °C/ 8.23 min, respectively. We note that the retention time tE is directly proportional to K calculated in the computational studies. 

TCAP and tE,CAP of Ls-PS and Lu-PS show a small but measurable difference reflecting the stronger interaction of the unsaturated DPE than saturated DPE with the C18 stationary phase although the difference is not large enough to change the LCCC elution sequence of ring PS and linear PS. Ring PS exhibits significantly higher TCAP and longer tE,CAP than both linear PS samples. This difference can be attributed mainly to the 

Figure 8. Chromatograms of Ls-PS, Lu-PS, and Ring-PS at their respective CAP: (a) CAP of Ls-PS; TCAP = 14.8 °C, tE,CAP = 7.4 min. (b) CAP of Lu-PS; TCAP = 14.95 °C, tE,CAP = 7.6 min. (c) CAP of Ring-PS; TCAP = 17.3 °C, tE,CAP = 8.23 min. tE,CAP of each polymer is marked with a vertical dashed line in the respective chromatogram. 

8785 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

Macromolecules 

Article 

Figure 9. Log MW vs elution time plot of Ls-PS and Ring-PS at (a) TCAP(Ls-PS) and at (b) TCAP(Ring-PS). 

architectural difference, i.e., linear versus ring chains, not to the functionality difference. These experimental results are in good agreement with the Monte Carlo simulation results where |εw(CAP, ring)| < |εw(CAP, linear)| and K(CAP, ring) > K(CAP, linear). The LCCC separation of the ring products at the CAP of linear polymers such as those shown in Figure 8a has been a typical method to fractionate Ring-PS from Ls-PS after the ringclosure reaction. Often in these applications, the end-group effect was ignored, and the CAP for linear polymers was determined using polymer standard samples lacking the end functional groups of the precursors. The linear polymer precursors with unreacted end functional groups such as in Lu-PS might elute later than the linear PS standard and become poorly resolved from the ring products. Also, the separation method works well for the high-MW polymers, but the resolution becomes poorer as MW decreases. In particular, if the end group is less favorable for the LCCC separation such as in Lu-PS, LCCC fractionation of Ring-PS from linear precursor would not be satisfactory for lowMW polymers. On the other hand, if we choose the chromatography condition at the CAP of Ring-PS for the LCCC fractionation instead of the CAP of linear polymers, a better separation of low-MW Ring-PS from linear precursor would be possible (see Figure 8c). 

Figure 9 displays plots of log MW vs elution time (peak position) of Ls-PS and Ring-PS at the CAP of each PS, respectively. Ring-PSs elute in the increasing order of MW (IC mode) at TCAP(Ls-PS) while Ls-PSs elute in the decreasing order of MW (SEC mode) at TCAP(Ring-PS). Since TCAP(Ls-PS) < TCAP(Ring-PS), in the LCCC separation of Ring-PS at TCAP(LsPS), Ring-PS should be in IC mode while, in the LCCC separation of Ls-PS at TCAP(Ring-PS), Ls-PS is in SEC mode. This is also consistent with the Monte Carlo simulation results shown in Figure 3. 

## IV. SUMMARY AND CONCLUSIONS 

This joint computational and experimental study examined how the CAP of the ring polymer compares with the CAP of the linear polymer. The CAP is defined as the coelution point of polymers with the same architecture but with different molecular weights. It was shown that TCAP(ring) > TCAP(linear) as long as there are no other complications. This implies that the adsorption transition for ring polymers occurs at higher temperature (less attractive segment/surface interaction) than for linear polymers. Results presented here are also in agreement with the earlier studies in which rings were found to be slightly more adsorbed than the linear polymers.[28][,][30] However, there were other studies reported that linear polymers are more absorbed than the rings especially for high molecular weight polymers. We note that the 

CAP identified using the coelution method (least dependence of K on chain length) corresponds to the usual understanding of adsorption transition best. When the segment/surface interaction is stronger than the identified εw(CAP), the density profile of the chain inside the slit shows enhanced accumulation near the wall; when it is weaker than εw(CAP), the density profile shows depletion behavior near the wall. On the other hand, if one identifies the transition using the peak position in the plot of heat capacity versus |εw| (the commonly used approach in discussing transition in theoretical physics), the peak occurred at much higher value than |εw(CAP)| (see Figure S8). Moreover, the peak position identified this way did not show noticeable difference between ring and linear species. It seems that the peak position identified in the plot of heat capacity versus |εw| corresponds to a different transition, potentially the flattening of the chain on the surface, not the on and off adsorption transition. These data might explain why some other theoretical/computational studies reported that the CAP for ring and linear polymers are the same.[35][,][59] Another point to note is that the CAP identified here used chain length ranged from N = 20−100. Our earlier studies on the CAP of linear chains have included chain length up to N = 200. Nevertheless, we acknowledge that the CAP identified here is for finite chain length. 

For those interested in practical applications, this study shows that one can find the coelution point for ring polymers just like one finds the coelution point for linear polymers. In principle, one may tune the chromatography condition at the CAP of ring polymers and achieve separation of ring products from the linear precursors. However, it might be difficult to actually implement this approach since standards of ring polymers with different molecular weights are not easily available. This study further shows that linear polymer precursors with interactive end-groups may elute differently than the linear polymer standards. In the current experimental settings, linear polymer precursors (Lu-PS) interact more strongly than linear polymers lacking the double bond (Ls-PS), and hence Lu-PS elutes after Ls-PS. However, this elution order may change depending on experimental conditions. In the current computational study, we did not investigate how the existence of functional groups on the ends and on the ring might impact the relative chromatography elution of ring and linear polymers. This will be examined in a future study. 

## ■[APPENDIX. HISTOGRAM REWEIGHTING METHOD] APPLIED 

The histogram reweighting method[53][,][54][,][60][,][61] has been reported by others, and here, we give a simple overview of how it is applied 

8786 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

Macromolecules 

Article 

to twinbox simulations to obtain the partition coefficient K at any desired temperature. 

We perform regular canonical Monte Carlo simulations in the twinbox at discrete |εw| values, which can be interpreted as simulations at discrete βi (the parameter β = |εw|), βi = 1/kBTi, kB is the Boltzmann constant, Ti is the temperature, i = 1, 2, ..., R, and R is the number of runs. For each run, we collect a probability distribution f i(ϕI,ϕb,E;βi), where ϕI and ϕb are the polymer concentrations in the interior pore region and bulk region, respectively, and E is total energy of system. These probability distributions are combined according to the method suggested by Ferrenberg and Swendsen[60][,][61] to obtain the composite probability 

**==> picture [202 x 32] intentionally omitted <==**

where Ki is the total number of observations made for run i. The constant Ci is obtained by iteration from the relationship 

**==> picture [149 x 25] intentionally omitted <==**

Given an initial guess for the set of weights Ci, these two equations can be iterated until convergence. Once the composite probability is obtained, we can then calculate ⟨ϕI⟩ and ⟨ϕb⟩ at any β according to 

**==> picture [133 x 58] intentionally omitted <==**

The partition coefficient K at any β is given by K = ⟨ϕI⟩/⟨ϕb⟩. We have also tried to use histogram reweighting by building probability distribution f i(Ki,E;βi), where Ki = ϕI/ϕb are obtained during any observations. The final average value of K calculated this way however does not match the K calculated by obtaining ⟨ϕI⟩ and ⟨ϕb⟩ individually. Statistically speaking, these are different values, the former K = ⟨ϕI⟩/⟨ϕb⟩ while the latter K is equivalent to K = ⟨ϕI/ϕb⟩. A final comment is that the composite probability distribution φ(ϕI,ϕb,E;β) could be stored as two separate probability distributions: φ1(ϕI,E;β), φ2(ϕb,E;β). The histogram reweighting approach is applied on one set of probability f i(ϕI,E;βi) and then applied to another separate set f i(ϕb,E;βi). The two distributions however converge with the same sets of constant Ci. 

## ■[ASSOCIATED CONTENT] 

## *S Supporting Information 

The Supporting Information is available free of charge on the ACS Publications website at DOI: 10.1021/acs.macromol.6b01925. 

Additional information on the comparison of partition coefficients obtained via twinbox simulation and biased insertion method; chromatograms of LCCC fractions of cyclic polystyrene and linear polystyrene samples as well as the chemical structures of Lu-PS, R-PS, and Ls-PS (PDF) 

## ■[AUTHOR INFORMATION] 

## Corresponding Authors 

- *E-mail ywang@memphis.edu (Y.W.). 

*E-mail tc@postech.ac.kr (T.C.). 

## Author Contributions 

J.D.Z. and Y.J. made equal contributions. 

## Notes 

The authors declare no competing financial interest. 

## ■[ACKNOWLEDGMENTS] 

T.C. acknowledges support from NRF-Korea (2015R1A2A2A01004974). Y.W. acknowledges support from NSF Tennessee EPSCOR funding (grant EPS-1004083) and the University of Memphis Honors Program that supports A.G. 

## ■[REFERENCES] 

(1) Fleer, G. J.; Cohen Stuart, M. A.; Scheutjens, J. M. H. M.; Cosgrove, T.; Vincent, B. Polymers at Interfaces; Chapman & Hall: London, UK, 1993. 

(2) Birshtein, T. M. Theory of Adsorption of Macromolecules. 1. The Desorption-Adsorption Transition Point. Macromolecules 1979, 12 (4), 715−721. 

(3) Birshtein, T. M.; Zhulina, E. B.; Skvortsov, A. M. Adsorption of Polypeptides on Solid Surfaces. I. Effect of Chain Stiffness. Biopolymers 1979, 18 (5), 1171−1186. 

(4) Gorbunov, A. A.; Zhulina, E. B.; Skvortsov, A. M. Theory of Adsorption of Macromolecules in Cylindrical Pores and at Surfaces of Cylindrical Shape. Polymer 1982, 23 (8), 1133−1142. 

(5) Hammersley, J. M.; Torrie, G. M.; Whittington, S. G. Self-Avoiding Walks Interacting with a Surface. J. Phys. A: Math. Gen. 1982, 15 (2), 539−571. 

(6) Eisenriegler, E.; Kremer, K.; Binder, K. Adsorption of PolymerChains at Surfaces: Scaling and Monte Carlo Analysis. J. Chem. Phys. 1982, 77 (12), 6296−6320. 

(7) Birshtein, T. M. Theory of Adsorption of Macromolecules. 2. Phase Transitions in Adsorption: General Approach. Macromolecules 1983, 16 (1), 45−50. 

(8) Descas, R.; Sommer, J.-U.; Blumen, A. Static and Dynamic Properties of Tethered Chains at Adsorbing Surfaces: A Monte Carlo Study. J. Chem. Phys. 2004, 120 (18), 8831. 

(9) Metzger, S.; Muller, M.; Binder, K.; Baschnagel, J. Adsorption Transition of a Polymer Chain at a Weakly Attractive Surface: Monte Carlo Simulation of Off-Lattice Models. Macromol. Theory Simul. 2002, 11 (9), 985−995. 

(10) Pasch, H. Analysis of Complex Polymers by Interaction Chromatography. Adv. Polym. Sci. 1997, 128, 1−45. 

(11) Chang, T. Recent Advances in Liquid Chromatography Analysis of Synthetic Polymers. Adv. Polym. Sci. 2003, 163, 1−60. 

(12) Chang, T. Polymer Characterization by Interaction Chromatography. J. Polym. Sci., Part B: Polym. Phys. 2005, 43 (13), 1591−1607. 

(13) Gorbunov, A. A.; Skvortsov, A. M. Statistical Properties of Confined Macromolecules. Adv. Colloid Interface Sci. 1995, 62 (1), 31− 108. 

(14) Ziebarth, J. D.; Wang, Y. Interactions of Complex Polymers with Nanoporous Substrate. Soft Matter 2016, 12, 5245−5256. 

(15) Radke, W. Polymer Separations by Liquid Interaction Chromatography: Principles - Prospects - Limitations. J. Chromatogr. A 2014, 1335, 62−79. 

(16) Brun, Y. The Mechanism of Copolymer Retention in Interactive Polymer Chromatography. I. Critical Point of Adsorption for Statistical Copolymers. J. Liq. Chromatogr. Relat. Technol. 1999, 22 (20), 3027− 3065. 

(17) Brun, Y. The Mechanism of Copolymer Retention in Interactive Polymer Chromatography. II. Gradient Separation. J. Liq. Chromatogr. Relat. Technol. 1999, 22 (20), 3067−3090. 

(18) Skvortsov, A. M.; Gorvunov, A. A. Adsorption Effects in the Chromatography of Polymers. J. Chromatogr. A 1986, 358 (1), 77−83. 

(19) Guttman, C. M.; Di Marzio, E. A.; Douglas, J. F. Influence of Polymer Architecture and Polymer-Surface Interaction on the Elution Chromatography of Macromolecules through a Microporous Media. Macromolecules 1996, 29 (17), 5723−5733. 

8787 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

Macromolecules 

Article 

(20) Ziebarth, J. D.; Wang, Y.; Polotsky, A.; Luo, M. Dependence of the Critical Adsorption Point on Surface and Sequence Disorders for SelfAvoiding Walks Interacting with a Planar Surface. Macromolecules 2007, 40 (9), 3498−3504. 

(21) Gong, Y.; Wang, Y. Partitioning of Polymers into Pores near the Critical Adsorption Point. Macromolecules 2002, 35, 7492−7498. (22) Narros, A.; Moreno, A. J.; Likos, C. N. Effects of Knots on Ring Polymers in Solvents of Varying Quality. Macromolecules 2013, 46 (9), 3654−3668. (23) Takano, A.; Ohta, Y.; Masuoka, K.; Matsubara, K.; Nakano, T.; Hieno, A.; Itakura, M.; Takahashi, K.; Kinugasa, S.; Kawaguchi, D.; et al. Radii of Gyration of Ring-Shaped Polystyrenes with High Purity in Dilute Solutions. Macromolecules 2012, 45 (1), 369−373. (24) Gooßen, S.; Bras, A. R.; Pyckhout-Hintzen, W.; Wischnewski, A.; Richter, D.; Rubinstein, M.; Roovers, J.; Lutz, P. J.; Jeong, Y.; Chang, T.; et al. Influence of the Solvent Quality on Ring Polymer Dimensions. Macromolecules 2015, 48 (5), 1598−1605. (25) Kapnistos, M.; Lang, M.; Vlassopoulos, D.; Pyckhout-Hintzen, W.; Richter, D.; Cho, D.; Chang, T.; Rubinstein, M. Unexpected PowerLaw Stress Relaxation of Entangled Ring Polymers. Nat. Mater. 2008, 7 (12), 997−1002. 

(26) Santangelo, P. G.; Roland, C. M.; Chang, T.; Cho, D.; Roovers, J. Dynamics near the Glass Temperature of Low Molecular Weight Cyclic Polystyrene. Macromolecules 2001, 34 (26), 9002−9005. (27) Roovers, J.; Toporowski, P. M. Synthesis and Characterization of Ring Polybutadienes. J. Polym. Sci., Part B: Polym. Phys. 1988, 26 (6), 1251−1259. (28) Van Lent, B.; Scheutjens, J.; Cosgrove, T. Self-Consistent Field Theory for the Adsorption of Ring Polymers from Solution. Macromolecules 1987, 20 (2), 366−370. (29) Stratouras, G.; Kosmas, M. Are Ring Polymers Adsorbed on a Surface More than Linear Polymers? Macromolecules 1992, 25 (12), 3307−3308. (30) Kosmas, M. K. Ideal Polymer Chains of Various Architectures at a Surface. Macromolecules 1990, 23 (7), 2061−2065. (31) Micheletti, C.; Orlandini, E. Numerical Study of Linear and Circular Model DNA Chains Confined in a Slit: Metric and Topological Properties. Macromolecules 2012, 45 (4), 2113−2121. (32) Li, B.; Sun, Z.-Y.; An, L.-J.; Wang, Z.-G. Influence of Topology on the Free Energy and Metric Properties of an Ideal Ring Polymer Confined in a Slit. Macromolecules 2015, 48 (23), 8675−8680. (33) Benkova, Z.; Cifra, P. Comparison of Linear and Ring DNA Macromolecules Moderately and Strongly Confined in Nanochannels. Biochem. Soc. Trans. 2013, 41 (2), 625−629. 

(34) Benkova, Z.; Cifra, P. Simulation of Semiflexible Cyclic and Linear Chains Moderately and Strongly Confined in Nanochannels. Macromolecules 2012, 45 (5), 2597−2608. 

(35) Sheng, J.; Luo, K. Conformation and Adsorption Transition on an Attractive Surface of a Ring Polymer in Solution. RSC Adv. 2015, 5 (3), 2056−2061. 

(36) Minina, E.; Arnold, A. Entropic Segregation of Ring Polymers in Cylindrical Confinement. Macromolecules 2015, 48 (14), 4998−5005. (37) Clarson, S.; Semlyen, J. Cyclic Polysiloxanes: 1. Preparation and Characterization of Poly(phenylmethylsiloxane). Polymer 1986, 27 (10), 1633−1636. 

(38) Geiser, D.; Hocker, H. Synthesis and Investigation of Macrocyclic Polystyrene. Macromolecules 1980, 13 (3), 653−656. 

(39) Roovers, J.; Toporowski, P. M. Synthesis of High Molecular Weight Ring Polystyrenes. Macromolecules 1983, 16 (6), 843−849. (40) Hild, G.; Strazielle, C.; Rempp, P. Cyclic Macromolecules. Synthesis and Characterization of Ring-Shaped Polystyrenes. Eur. Polym. J. 1983, 19 (8), 721−727. 

(41) Takano, A.; Kushida, Y.; Aoki, K.; Masuoka, K.; Hayashida, K.; Cho, D.; Kawaguchi, D.; Matsushita, Y. HPLC Characterization of Cyclization Reaction Product Obtained by End-to-End Ring Closure Reaction of a Telechelic Polystyrene. Macromolecules 2007, 40 (3), 679−681. 

(42) Lepoittevin, B.; Dourges, M.-A.; Masure, M.; Hemery, P.; Baran, K.; Cramail, H. Synthesis and Characterization of Ring-Shaped Polystyrenes. Macromolecules 2000, 33 (22), 8218−8224. 

(43) Cho, D.; Park, S.; Kwon, K.; Chang, T.; Roovers, J. Structural Characterization of Ring Polystyrene by Liquid Chromatography at the Critical Condition and MALDI−TOF Mass Spectrometry. Macromolecules 2001, 34 (21), 7570−7572. 

(44) Lee, W.; Lee, H. C.; Cho, D.; Chang, T.; Gorbunov, A. A.; Roovers, J. Retention Behavior of Linear and Ring Polystyrene at the Chromatographic Critical Condition. Macromolecules 2002, 35 (2), 529−538. 

(45) Lee, H. C.; Lee, H.; Lee, W.; Chang, T. H.; Roovers, J. Fractionation of Cyclic Polystyrene from Linear Precursor by HPLC at the Chromatographic Critical Condition. Macromolecules 2000, 33 (22), 8119−8121. 

(46) Elupula, R.; Oh, J.; Haque, F. M.; Chang, T.; Grayson, S. M. Determining the Origins of Impurities during Azide−Alkyne Click Cyclization of Polystyrene. Macromolecules 2016, 49 (11), 4369−4372. 

(47) Gorshkov, A. V.; Much, H.; Becker, H.; Pasch, H.; Evreinov, V. V.; Entelis, S. G. Chromatographic Investigations of Macromolecules in the “critical Range” of Liquid Chromatography. J. Chromatogr. A 1990, 523, 91−102. 

(48) Baran, K.; Laugier, S.; Cramail, H. Fractionation of Functional Polystyrenes, Poly(ethylene Oxide)s and Poly(styrene)-B-Poly(ethylene Oxide) by Liquid Chromatography at the ExclusionAdsorption Transition Point. J. Chromatogr., Biomed. Appl. 2001, 753 (1), 139−149. 

(49) Im, K.; Kim, Y.; Chang, T.; Lee, K.; Choi, N. Separation of Branched Polystyrene by Comprehensive Two-Dimensional Liquid Chromatography. J. Chromatogr. A 2006, 1103 (2), 235−242. 

(50) Wang, Y. M.; Teraoka, I. Computer Simulation of Semidilute Polymer Solutions in Confined Geometry: Pore as a Microscopic Probe. Macromolecules 1997, 30 (26), 8473−8477. 

(51) Wang, Y.; Teraoka, I.; Cifra, P. Lattice Monte Carlo Simulation for the Partitioning of a Bimodal Polymer Mixture into a Slit. Macromolecules 2001, 34 (1), 127−133. 

(52) Wang, X.; Lísal, M.; Prochazka, K.; Limpouchova, Z. Computer Study of Chromatographic Separation Process: A Monte Carlo Study of H-Shaped and Linear Homopolymers in Good Solvent. Macromolecules 2016, 49 (3), 1093−1102. 

(53) Roux, B. The Calculation of the Potential of Mean Force Using Computer Simulations. Comput. Phys. Commun. 1995, 91 (1−3), 275− 282. 

(54) Panagiotopoulos, A. Z. Monte Carlo Methods for Phase Equilibria of Fluids. J. Phys.: Condens. Matter 2000, 12 (3), R25−R52. 

(55) Orelli, S.; Jiang, W.; Wang, Y. A Computational Investigation of the Critical Condition Used in the Liquid Chromatography of Polymers. Macromolecules 2004, 37 (26), 10073−10078. 

(56) Jiang, W.; Khan, S.; Wang, Y. Retention Behaviors of Block Copolymers in Liquid Chromatography at the Critical Condition. Macromolecules 2005, 38 (17), 7514−7520. 

(57) Higashihara, T.; Nagura, M.; Inoue, K.; Haraguchi, N.; Hirao, A. Successive Synthesis of Well-Defined Star-Branched Polymers by a New Iterative Approach Involving Coupling and Transformation Reactions. Macromolecules 2005, 38 (11), 4577−4587. 

(58) Gorbunov, A. A.; Vakhrushev, A. V. Theory of Chromatography of Linear and Cyclic Polymers with Functional Groups. Polymer 2004, 45 (21), 7303−7315. 

(59) Soteros, C. E. Adsorption of Uniform Lattice Animals with Specified Topology. J. Phys. A: Math. Gen. 1992, 25 (11), 3153−3173. (60) Ferrenberg, A. M.; Swendsen, R. H. Optimized Monte Carlo Data Analysis. Phys. Rev. Lett. 1989, 63 (12), 1195−1198. 

(61) Ferrenberg, A. M.; Swendsen, R. H. New Monte Carlo Technique for Studying Phase Transitions. Phys. Rev. Lett. 1988, 61 (23), 2635− 2638. 

8788 

DOI: 10.1021/acs.macromol.6b01925 Macromolecules 2016, 49, 8780−8788 

