using apicalls;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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

namespace testDesktopApp
{
    /// <summary>
    /// Interaction logic for editWindow.xaml
    /// </summary>
    public partial class editWindow : Window
    {
        private ObservableCollection<CompanyItem> _companyList;
        public editWindow(Company company, ObservableCollection<CompanyItem> companyList)
        {
            InitializeComponent();
            _companyList = companyList;
        }
        private void textChangeObec(object sender, TextChangedEventArgs e)
        {
            if (ObchodniJmenoTextBox.Text.Length == 0)
            {
                SaveButton.IsEnabled = false;
            }
            else
            {
                SaveButton.IsEnabled = true;
            }
        }

        private void textChangeObchodniJmeno(object sender, TextChangedEventArgs e)
        {
            if (SidloTextBox.Text.Length == 0)
            {
                SaveButton.IsEnabled = false;
            }
            else
            {
                SaveButton.IsEnabled = true;
            }
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            _companyList.Add(new CompanyItem
            {
                obchodniJmeno = ObchodniJmenoTextBox.Text,
                nazevObce = SidloTextBox.Text,
                ico = IcoTextBox.Text,
                dic = DicTextBox.Text,
                poznamka = PoznamkaTextBox.Text
            });
            this.Close();
        }

        private void back_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}
