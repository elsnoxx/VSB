using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace reseniWPF_1
{
    /// <summary>
    /// Interaction logic for Window1.xaml
    /// </summary>
    public partial class Window1 : Window
    {
        public Window1(JsonData result)
        {
            InitializeComponent();
            InitData(result);
        }

        public void InitData(JsonData result)
        {
            InputBoxDIC.Text = result.dic;
            InputBoxNazev.Text = result.obchodniJmeno;
            InputBoxObec.Text = result.sidlo.nazevObce;
        }

        public void SubmitButton_Click(object sender, RoutedEventArgs e)
        {
            
            string dic = InputBoxDIC.Text;
            string nazev = InputBoxNazev.Text;
            string obec = InputBoxObec.Text;
            if (string.IsNullOrEmpty(nazev))
            {
                ErrorText.Text = "Zadej nazev";
                ErrorText.Visibility = Visibility.Visible;
            }
            else if (string.IsNullOrEmpty(obec))
            {
                ErrorText.Text = "Zadej obec";
                ErrorText.Visibility = Visibility.Visible;
            }
            // Perform any actions you need with the input values
            MessageBox.Show($"DIC: {dic}, Nazev: {nazev}, Obec: {obec}");
        }
    }
}
