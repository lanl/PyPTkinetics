# PyPTkinetics

O# (O4940)</br>
doi: [10.11578/dc.20250822.5](https://doi.org/10.11578/dc.20250822.5)</br>

This basic research code computes the time-dependent phase transition between two solid crystalline phases under dynamic conditions using a microstructure-aware model described in https://www.arxiv.org/abs/2504.00250 (LA-UR-24-32576) as well as effective phenomenological models which can capture the main features of the microscopic model. Analytic equations of state are included for metals such as alpha / epsilon iron and beta / gamma tin. Results, such a volume fraction of the new phase as a function of pressure, are visualized using matplotlib.
 
 © 2025. Triad National Security, LLC. All rights reserved.

## Author

Daniel N. Blaschke


This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

## Requirements

* Python >=3.9,</br>
* [numpy](https://numpy.org/doc/stable/user/) >=1.19,</br>
* [scipy](https://docs.scipy.org/doc/scipy/reference/) >=1.9,</br>
* [matplotlib](https://matplotlib.org/) >=3.3</br>
* [pandas](https://pandas.pydata.org/) >=1.3 (and Jinja2)</br>


### Optional but recommended:
* [PyDislocDyn](https://github.com/dblaschke-LANL/PyDislocDyn) and its dependencies

## License

This program is Open-Source under the BSD-3 License.</br>
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:</br>
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.</br>
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.</br>
* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.</br>
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
