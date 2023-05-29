
import streamlit as st
import pandas as pd
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d,plot3d_parametric_line
import plotly as ply
import plotly.express as ex
import plotly.graph_objects as gro
from plotly.subplots import make_subplots
tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplos","Aplicaciones"])
with tab1:
         st.title(":blue[Metodo de Newton(-Raphson)]")
         st.header("Teorema")
         """
         El metodo de Newton es una alternativa del metodo del punto 
         fijo ya que la gran desventaja de este se consiste en obtener ecuaciones $g_i$ que converja 
         en el intervalo de solucion y dicha búsqueda puede representar un esfuerzo de cálculo considerable, en lo unico en que se parecen en estos dos metodos es 
         la generalización multivariable del método de Newtoon para una sola variable.
         Este metodo solo sirve para los sistemas de ecuaciones no lineales, este requiere algunos conceptos previos, los cuales se presentan a continuación.
         La derivada, cuando se trabaja con funciones con varias variables, se emplean las derivadas parciales. La generalización de derivada para sistemas de funciones 
         de varias variables es la matriz jacobiana. \\
         :red[**Definición**]. Matriz jacobiana: Sean $f_i (x_1, x_2, ..., x_n)$ con $ 1 \leq i \leq n $, funciones con n variables $(x_i)$ independientes. La matriz jacobiana $J(x_1, x_2,..., x_n)$
         está dada por las derivadas parciales de cada una de las funciones con respecto a cada una de las variables: 
         """
         "Matriz Jacobiana (J):"
         st.latex(r"""
                  \begin{bmatrix}
                  \dfrac{\partial f_1}{\partial x_1} (x_0, x_1, ..., x_n) & \dfrac{\partial f_1}{\partial x_2} (x_0, x_1, ..., x_n)&... ... & \dfrac{\partial f_1}{\partial x_n} (x_0, x_1, ..., x_n) \\
                  & & &\\
                  \dfrac{\partial f_2}{\partial x_1} (x_0, x_1, ..., x_n) & \dfrac{\partial f_2}{\partial x_2} (x_0, x_1, ..., x_n)&... ...  &\dfrac{\partial f_2}{\partial x_n} (x_0, x_1, ..., x_n) \\
                  ... & & & \\
                  \dfrac{\partial f_n}{\partial x_1} (x_0, x_1, ..., x_n) & \dfrac{\partial f_n}{\partial x_2} (x_0, x_1, ..., x_n)&... ...& \dfrac{\partial f_n}{\partial x_n} (x_0, x_1, ..., x_n)
                  \end{bmatrix}
                  """)
         st.latex(r"""
                  \begin{bmatrix} 
                  f_{1 x_1} & f_{1 x_2}&... ... & f_{1 x_n} \\
                  & & &\\
                  f_{2 x_1} & f_{2 x_1}&... ...  &f_{2 x_n} \\
                  ... & & & \\
                  f_{n x_1} & f_{n x_2}&... ...& f_{n x_n}
                  \end{bmatrix}
                  """)
         """
         El método de Newton para una variable se basa en la expansión de la serie de Taylor de primer orden:\\
         $f(x_{i+1}) = f(x_i) + (x_{i+1} - x_i) f'(x_i)$\\
         Donde $x_i$ es el valor inicial de la raíz y $x_{i+1}$ es el punto en el cual la derivada (pendiente) interseca del eje. En esta intersección 
         $f(x_{i+1})$ por definición es igual a cero y la forma iterativa del método puede escribirse como:
         """
         """   $x_{i+1} = x_i - \dfrac{f(x_i)}{f'(x_i)}$ """
         """
         La forma 'simple' de la ecuación del método de Newton.
         La forma para varias ecuaciones se deriva en forma idéntica, pero a partir de la serie de Taylor para varias variables.
         Escribiendo esta ecuación en forma matricial: 
                   + Para el sistema $f_n(x_1, x_2,... x_n)$ se tiene: $F(X)=0$
                   + Definiendo los vectores columna como \\
                   $ F = (f_{1}, f_{2}, ..., f_{n})^{t},$ $ X = (x_{1}, x_{2}, x_{n})^{t}$
                   + La extensión del método de Newton para sistemas no lineales es : \\
                   $X^{(k+1)} = X^{(k)} - [F'(X^{(k)}]^{-1} F(X^{(k)})$ \\
                   Donde $F'(x^ {(k)})$ es la matriz jacobiana: \\
                   $X^{(k+1)} = X^{(K)} - J(X^{(k)})^{-1} F(X^{(k)})$ \\
                   Esta ecuación es análoga a la del método de Newton-Raphson para una ecuación, sólo que la derivada aparece en el numerador como la inversa de la matriz jacobiana.
                   La ecuacion $X^{(k+1)} = X^{(K)} - J(X^{(k)})^{-1} F(X^{(k)})$ es la representación del método de Newton-Raphson para sistemas de no lineales y generalmente se espera que 
                   dé una convergencia cuadratica, siempre y cuando se conozca un valor inicial suficiententemente preciso y exista $J(X^{k})^{-1}$ \\
                   
                   """
         with tab2: 
                  st.title(":blue[Ejercicio:]")
                  """
                   Sea el siguiente sistema de ecuaciones: \\
                           + $f_{1} (x,y) = 4-x^{2} -y^{2} = 0 $ \\
                           + $f_{2} (x,y) = 1-e^{x} -y = 0 $
                  """
                  """
                  Se obtienen las derivadas parciales para escribir la matriz jacobiana: 
                  """
                  st.latex(r"""
                  \begin{matrix} 
                  f_{1x} = -2x & f_{1y} = -2y \\
                  & \\
                  f_{2x} = -e^{x} &  f_{2y} = -1 
               
                  \end{matrix}
                  """)
                  """
                  Jacobiana J(X)
                  """
                  st.latex(r"""
                  \begin{matrix} 
                  f_{1x} = -2x & f_{1y} = -2y \\
                  & \\
                  f_{2x} = -e^{x} &  f_{2y} = -1 
               
                  \end{matrix}
                  """)
                  """
                  Se eligen valores iniciales para la aproximación $X^{(0)} = (1,-1.7)^{t}$ , como valor de $x_0$ y $y_0$ que se sustituyen en $X^{(k+1)} = X^{(K)} - J(X^{(k)})^{-1} F(X^{(k)})$
                  """
                  st.latex(r""" X^{(1)} = 
                  \begin{bmatrix} 
                  1\\
                   \\
                  -1.7
                  \end{bmatrix}
                  -
                  \begin{bmatrix} 
                  -2 & 3.4 \\
                  & \\
                 -2.7183 &  -1.0 
               
                  \end{bmatrix} ^{-1}
                  
                  \begin{bmatrix} 
                  0.1100\\
                   \\
                  -0.0183
                  \end{bmatrix}
                  =
                  \begin{bmatrix} 
                  1.00426\\
                   \\
                  -1.72985
                  \end{bmatrix}
                  """)
                  """
                  Estos valores tienen una aproximacion optima pero aun se puede mejorar, para esto requiere otra iteracion para mejorarla. 
                  """
                  st.latex(r""" X^{(2)} = 
                  \begin{bmatrix} 
                  1.00426\\
                   \\
                  -1.72985
                  \end{bmatrix}
                  -
                  \begin{bmatrix} 
                  -2.00851 & 3.4597 \\
                  & \\
                 -2.72987 &  -1.0 
               
                  \end{bmatrix} ^{-1}
                  
                  \begin{bmatrix} 
                  -0.0091\\
                   \\
                  -0.00002
                  \end{bmatrix}
                  =
                  \begin{bmatrix} 
                  1.00417\\
                   \\
                  -1.72963
                  \end{bmatrix}
                  """)
                  """
                  Las iteraciones se detienen cuando la norma espectral del vector $F(X)$ es menor a una tolerancia previamente establecida. \\
                  \\
                  Si se cambia el punto inicial a $X^{(0)} = (-1.5, 0.6)^{t}$ se obtiene la segunda raíz $X=(-1.816264, 0.837367) ^{t}$.\\
                  \\
                  Es importante calcular la matriz jacobiana y su inversa en cada interación. 
                  """
         with tab3:
             def parse_inputsys(inp):
                 eq = []
                 sfix = ''
                 for i in inp:
                     if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
                         sfix = sfix + i
                         for t in sfix.split(','):
                             eq.append(sy.parse_expr(t,transformations='all'))
                             return eq
             def get_maxsymbs(sys):
                maxs = 0
                s = []
                for i in sys:
                    if max(maxs,len(i.free_symbols)) != maxs:
                         s = list(i.free_symbols)
                
                return s
            
             def jacobian(ff,symb):
                """
                Toma un vector de funciones y un vector de símbolos y devuelve la matriz jacobiana de las funciones con respecto a
                los símbolos
                :param ff: la función
                :param symb: los símbolos que se utilizan en la función
                :return: Una matriz de las derivadas parciales de la función con respecto a las variables.
                """
                m = []
                
                for i in range(0, len(ff)):
                    aus = []
                    for j in range(0, len(symb)):
                         aux.append(sy.diff(ff[i],symb[j]))
                    m.append(aux)
                
                return np.array(m)
             def hessian(ff,symb):
                 """Toma un vector de funciones y un vector de símbolos y devuelve la matriz hessiana de las funciones con respecto a
                    los símbolos
                    :param ff: una lista de funciones de la forma f(x,y,z)
                    :param symb: los símbolos que se utilizan en la función
                    :return: Una matriz de las segundas derivadas de la función.
    
                 """
                 m = []
                 for i in range(0, len(ff)):
                     aux = []
                     for j in range(0,len(symb)):
                         aux.append(sy.diff(ff[i],symb[j],2))
                     m.append(aux)
                 return np.array(m)
             def eval_matrix(matrix, v, sy, symb):
                 """
                 Toma una matriz, una lista de símbolos y una lista de valores, y devuelve la matriz con los símbolos sustituidos por los
                    valores

                    :param matriz: la matriz del sistema de ecuaciones
                    :param v: el vector de valores de las variables
                    :param symb: los símbolos que se utilizarán en la matriz
                    :return: la matriz con los valores de las variables sustituidos por los valores del vector v.
    
                 
                 """ 
                 e = 0
                 mm = []
                 for i in range(0,len(matrix)):
                      aux = []
                      ev = []
                      for k in range(0,len(symb)):
                          ev.append((symb[k],v[k]))
                      for j in range(len(matrix[i])):
                          aux.append(matrix[i][j].subs(ev).evalf())
                      mm.append(aux)
                 return np.array(mm)
             def avelVector(ff, x0,symb):
                 """Dada una lista de expresiones simbólicas, una lista de valores para los símbolos y una lista de los símbolos, evalúe las
    expresiones simbólicas con los valores dados

    :param ff: el vector de funciones
    :param x0: conjetura inicial
    :param symb: los símbolos que se utilizan en la expresión simbólica
    :return: el valor de la función en el punto x0.
    
                 """
                 v = []
                 for i in range(0,len(ff)):
                     ev = []
                     for k in range(0,len(x0)):
                         ev.append((symb[k],x0[k]))
                         
                     v.append(ff[i].subs(ev).elaf())
                 return np.array(v)
             def NewtonMethod(ff, x0, symb):
                 """
                  La función recibe un vector de funciones, un vector de conjeturas iniciales y un vector de símbolos. A continuación, calcula
    la matriz jacobiana, la matriz jacobiana evaluada en la conjetura inicial, la inversa de la matriz jacobiana evaluada en la conjetura inicial, el vector de funciones evaluado en la conjetura inicial, y luego el paso de Newton.
    evaluada en la conjetura inicial, el vector de funciones evaluadas en la conjetura inicial y el paso de Newton.

    La función devuelve el paso Newton.

    :param ff: la función de la que queremos encontrar la raíz
    :param x0: estimación inicial
    :param symb: los símbolos utilizados en la función
    :return: El valor de retorno es el valor x_np1.
    
                 """
                 j = jacobian(ff,symb)
                 jev = sy.Matrix( eval_matrix(j,x0,symb))
                 jinv = jev.inv()
                 ffev = sy.Matrix(evalVector(np.transpose(ff),x0,symb))
                 mm = sy.Matrix(jinv)*ffev
                 x_np1 = sy.Matrix(np.transpose(np.array(x0)))
                 return list(x_np1-mm)
             
             def norm_inf(x_0, x_1):
                 """
                 La función `norm_inf` toma dos vectores `x_0` y `x_1` y devuelve la máxima diferencia absoluta entre los dos
    vectores

    :param x_0: el vector inicial
    :param x_1: el vector de la iteración actual
    :devuelve: La máxima diferencia entre los dos vectores.
    
                 """
                 a = [abs(x_1[i]-x_0[i]) for i in range(len(x_0))]
                 return max(a)
             
             def newton_method(ff,x_0,symbs,error,maxiter):
                 """
                 Dada una función (x,y)$, un punto inicial $, y una lista de símbolos,
    la función devolverá el siguiente punto $ en la secuencia del método de Newton

    :param ff: función a minimizar
    :param x_0: punto de partida
    :param symbs: los símbolos que usamos en la función
    :return: el valor final de x_0, la lista de valores de x, y la lista de valores de y.
    
                 """
                 xs = []
                 ys = []
                 xns = [x_0]
                 erros = []
                 iterr = []
                 while rue and iterr < maxiter:
                     x_1 = NewtonMethod(ff,x_0,symbs)
                     ninf = norm_inf(x_0,x_1)
                     erros.append(ninf)
                     x_0 = list(x_1)
                     xns.append(tuple(x_0))
                     xs.append(x_0[0])
                     ys.append(x_0[1])
                     if ninf < error:
                         break
                     iterr = iterr+1
                 return xns,xs,ys,erros
             
             def get_sympy_subplots(plot:Plot):
                 """Toma un objeto plot y devuelve un objeto figure de matplotlib

    param gráfico: El objeto plot a renderizar
    :type gráfico: Gráfico
    :return: Un objeto de figura matplotlib.
    
                 """
                 
                 backend = MatplotlibBackend(plot)
                 backend.process_series()
                 backend.fig.tight_layout()
                 return backend.plt
             st.title(':blue[Metodo de Newton]')
             st.subheader(':blue[Descripción del método]')
             
             methd_desc1 = r'''
                    El método de Newton para sistemas de ecuaciones no lineales es una extensión del método utilizado para
                resolver una sola ecuación, por tanto sigue la misma estrategia que se empleó en el caso de una sola ecuación:
                linealizar y resolver, repitiendo los pasos con la frecuencia necesaria. Consideramos, por sencillez, el caso de
                dos ecuaciones con dos variables:
                '''
             methd_desc2 = r'''

                \begin{cases}
                f(x,y) = 0 \\
                g(x,y) = 0 \\
                \end{cases}
                '''
             methd_desc3 = r'''
                F'(x,y) = \begin{pmatrix}
                \frac{\partial f}{\partial x} (x,y) & \frac{\partial f}{\partial y} (x,y) \\
                \frac{\partial g}{\partial x} (x,y) & \frac{\partial g}{\partial y} (x,y)
                \end{pmatrix}
                '''
             methd_desc4 = r'''
             \begin{pmatrix}
                x_{n+1}\\
                y_{n+1}
                \end{pmatrix} =
                \begin{pmatrix}
                x_{n}\\
                y_{n}
                \end{pmatrix} - (F'(x_n,y_n))^{-1} F(x_n,y_n)
                '''
             st.write(methd_desc1)
             st.latex(methd_desc2)
             st.write('Si definimos la función $F(x, y) = (f (x, y), g(x, y))$, transformamos el sistema anterior en')
             st.latex('F(x,y) = 0')
             st.write(''' Podemos realizar entonces una formulación idéntica al caso de una variable donde la derivada está dada en
                este caso por la matriz jacobian''')
             st.latex(methd_desc3)
             st.write('''Así, el método de Newton comienza por un vector inicial $(x_0, y_0)^T$ y calcula el resto de aproximaciones
mediante ''')
             st.latex(methd_desc4)
             st.write('''
donde $(F'(x_n, y_n))^1$ es la matriz inversa de $F'(x_n, y_n)$. Para poder aplicar este método es necesario que
$F'(x, y)$ sea no singular.\\
Un problema del método anterior es que el cálculo de la matriz inversa es costoso computacionalmente y
debemos calcularla en cada paso. Esto se puede resolver descomponiendo el método en dos etapas:
''')
             st.write('1. Resolver el sistema lineal con dos ecuaciones y dos incognitas:')
             st.latex(r'''
                F'(x_n, y_n)
                \begin{pmatrix}
                u_{n}\\
                v_{n}
                \end{pmatrix} = -F(x_n, y_n)
                ''')
             st.write('2. Tomar como nueva aproximación:')
             st.latex(r'''
             \begin{pmatrix}
             x_{n+1}\\
             y_{n+1}
             \end{pmatrix} =
             \begin{pmatrix}
             x_{n}\\
             y_{n}
             \end{pmatrix} +
             \begin{pmatrix}
             u_{n}\\
             v_{n}
             \end{pmatrix}
             ''')
             st.subheader(':blue[Ejemplo]')
             st.subheader('Método')
             sys = st.text_input('Ingrese el sistema de ecuaciones ',value=r'{x^2+3*y*x+2,y**3*x**2-2*y**3-5}')
             try:
                 system = parse_inputsys(sys)
             except :
                 st.error('Error al introducir el sistema de ecuaciones', icon="🚨")
                 st.write('_El sistema de ecuaciones es:_')
                 psys ='F'+str(tuple(get_maxsymbs(system)))+ r'''
                     =\begin{cases}

                 '''
                 for i in system:
                     psys = psys + sy.latex(i)
                     psys = psys + r' = 0\\'

                 psys = psys + r'\end{cases}'

                 st.latex(psys)
                 fx = sy.lambdify(list(get_maxsymbs(system)),system[0])
                 fx2 = sy.lambdify(list(get_maxsymbs(system)),system[1])
                 try:
                     st.write('_Grafica del sistema como función implicita_ : ')
                     x,y = sy.symbols('x,y')
                     p1 = sy.plot_implicit(system[0],(x,-10,10),(y,-10,10),show=False,line_color='red')
                     for i in range(1,len(system)):
                         p1.append(sy.plot_implicit(system[i],(x,-10,10),(y,-10,10),show=False,line_color='blue')[0])
                     t2dplot =  get_sympy_subplots(p1)
                     st.pyplot(t2dplot)
                     st.write('_Grafica 3D del sistema_')
                     

                     plo1 = gro.Figure()

                     ran = np.linspace(-10,10,100)
                     su1 = [[fx(ran[xs],ran[ys]) for xs in range (0,len(ran)) ] for ys in range(0,len(ran))]
                     su2 = [[fx2(ran[xs],ran[ys]) for xs in range (0,len(ran)) ] for ys in range(0,len(ran))]

                     plo1.add_trace(gro.Surface(z=su1,name='Ecuación 1',opacity=.7))
                     plo1.add_trace(gro.Surface(z=su2,name='Ecuación 2',opacity=.7))

                     st.plotly_chart(plo1)
                     p13d = plot3d(system[0],(x,-10,10),(y,-10,10),show=False,surface_color='r')
                     for i in range(1,len(system)):
                         p13d.append(plot3d(system[i],(x,-10,10),(y,-10,10),show=False,surface_color='g')[0])

                     t3dplot = get_sympy_subplots(p13d)
                     st.pyplot(t3dplot)
                 except Exception as excep:
                     st.error(excep)
                     st.error('Error al graficar', icon="🚨")


                 initaprx = st.text_input('Ingrese una aproximacion inicial $x_0$: ',value=[-1,1])

                 intaprox = []
                 intstr = ''
                 for i in initaprx:

                     if i != '{' and i != '}' and i != '[' and i != ']' and i != ' ':
                         intstr = intstr + i

                 try:
                     st.write('La aproximacion inicial es: ')
                     intaprox = list(map(int, intstr.split(',')))
                     st.latex(sy.latex(sy.Matrix(list(intaprox))))
                 except:
                     st.error('Error al introducir la aproximación inicial', icon="🚨")
                 err = st.text_input('Ingrese el error de tolerancia: ',value='0.00001')
                 try:
                     st.write('El error de tolerancia es:', float(err))
                 except: 
                     st.error('Error al introducir el error de tolerancia', icon="🚨")
                
                 maxiter = st.slider('Maximo de Iteraciones',10,1000,10)
                 
                 st.write("_Matrix Jeacobiana_:")
                 symbs = (get_maxsymbs(system))
                 st.latex(sy.latex(sy.Matrix(jacobian(system,symbs))))
                 method = newton_method(system,list(intaprox),symbs,float(err),maxiter)
                 tabb = []
                 for i in range(0,len(method[1])):
                     aux = list(method[0][i])
                     aux.append(method[3][i])
                     tabb.append(aux)
                     
                 cols = list(map(str, list(symbs)))
                 cols.append("Error")
                 table = list(map(str, list(symbs)))
                 st.write(table)
                 try:
                     xs =[float(i) for i in method[1]]
                     xy =[float(i) for i in method[2]]
                     evalfx = [float(fx(i[0],i[1])) for i in method[0]]
                     
                     plo = gro.Figure()
                     plo.add_trace(gro.Scatter3d(x=xs,y=xy,z=evalfx,name='Aproximaciones'))

                     ranx = np.linspace(int(method[1][-1]-1),int(method[1][-1]+1),100)
                     rany = np.linspace(int(method[2][-1]-1),int(method[2][-1]+1),100)
                     su1 = [[fx(ranx[xs],rany[ys]) for xs in range (0,len(ranx)) ] for ys in range(0,len(rany))]
                     su2 = [[fx2(ranx[xs],rany[ys]) for xs in range (0,len(ranx)) ] for ys in range(0,len(rany))]
                     plo.add_trace(gro.Surface(z=su1,name='Ecuación 1',opacity=.8))
                     plo.add_trace(gro.Surface(z=su2,name='Ecuación 2',opacity=.7))

                     plo.update_layout(title='Grafica del Sistema')
                     st.plotly_chart(plo)
                     
                     
                     pf =get_sympy_subplots(p1)
                     pf.plot(method[1],method[2],'o')
                     st.pyplot(pf)
                     auxpd = plot3d(system[0],(x,-1+method[1][-1],method[1][-1]+1),(y,-1+method[2][-1],method[2][-1]+1),show=False,alpha=0.3,title='Grafica de la Solución')
                     auxpd.append(plot3d(system[1],(x,-1+method[1][-1],method[1][-1]+1),(y,-1+method[2][-1],method[2][-1]+1),show=False,alpha=0.5)[0])
                     pda = get_sympy_subplots(auxpd)
                     zs = []
                     for i in range(methd[1]):
                          zs.append(system[0].subs({x : method[1][i],y: method[2][i]}).evalf())
                     pda.plot(method[1],method[2],zs,'o',markersize=30,color='red')
                     st.pyplot(pda)
                 except Exception as e:
                     st.error(str(e))
                         






   


             

            
             
                
                          
                          
                          
                     
                         
                 
            
                     
             
             
            
                  
