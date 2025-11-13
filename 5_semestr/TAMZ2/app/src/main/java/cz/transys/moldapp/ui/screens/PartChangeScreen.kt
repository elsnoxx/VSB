package cz.transys.moldapp.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PartChangeScreen(onBack: () -> Unit) {
    // Carrier dropdown
    var expandedCarrier by remember { mutableStateOf(false) }
    var selectedCarrier by remember { mutableStateOf("") }
    val carrierList = listOf("#01", "#02", "#03", "#04")

    // Mold 1
    var mold1Type by remember { mutableStateOf("") }
    var mold1Code by remember { mutableStateOf("") }

    // Mold 2
    var mold2Type by remember { mutableStateOf("") }
    var mold2Code by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFECECEC))
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "🟠 Part Change",
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF1565C0),
            modifier = Modifier.padding(bottom = 20.dp)
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(16.dp)
        ) {

            // Carrier ComboBox
            ExposedDropdownMenuBox(
                expanded = expandedCarrier,
                onExpandedChange = { expandedCarrier = !expandedCarrier }
            ) {
                OutlinedTextField(
                    value = selectedCarrier,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Carrier") },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedCarrier)
                    },
                    modifier = Modifier
                        .menuAnchor()
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                )

                ExposedDropdownMenu(
                    expanded = expandedCarrier,
                    onDismissRequest = { expandedCarrier = false }
                ) {
                    carrierList.forEach { carrier ->
                        DropdownMenuItem(
                            text = { Text(carrier) },
                            onClick = {
                                selectedCarrier = carrier
                                expandedCarrier = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ---- Mold #1 ----
            Text(
                text = "Mold #1",
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                color = Color.DarkGray,
                modifier = Modifier.padding(vertical = 4.dp)
            )

            OutlinedTextField(
                value = mold1Type,
                onValueChange = { mold1Type = it },
                label = { Text("Type") },
                placeholder = { Text("např. NX4E FC LH - 08") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            )

            OutlinedTextField(
                value = mold1Code,
                onValueChange = { mold1Code = it },
                label = { Text("Mold ID") },
                placeholder = { Text("např. M88150N7001-08") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            )

            Spacer(modifier = Modifier.height(16.dp))

            // ---- Mold #2 ----
            Text(
                text = "Mold #2",
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                color = Color.DarkGray,
                modifier = Modifier.padding(vertical = 4.dp)
            )

            OutlinedTextField(
                value = mold2Type,
                onValueChange = { mold2Type = it },
                label = { Text("Type") },
                placeholder = { Text("např. NX4E FC RH - 08") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            )

            OutlinedTextField(
                value = mold2Code,
                onValueChange = { mold2Code = it },
                label = { Text("Mold ID") },
                placeholder = { Text("např. M88250N7001-08") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            )

            Spacer(modifier = Modifier.height(20.dp))

            // Submit button
            Button(
                onClick = {
                    // TODO: volání API (později)
                    // api.submitPartChange(selectedCarrier, mold1Type, mold1Code, mold2Type, mold2Code)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(60.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF1565C0),
                    contentColor = Color.White
                )
            ) {
                Text(
                    text = "SUBMIT",
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = onBack,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.Gray,
                    contentColor = Color.White
                )
            ) {
                Text("Zpět na menu")
            }
        }
    }
}