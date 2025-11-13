package cz.transys.moldapp.ui.screens

import android.util.Log
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
import cz.transys.moldapp.ui.apicalls.CarCodeList
import cz.transys.moldapp.ui.apicalls.MoldApiRepository
import cz.transys.moldapp.ui.apicalls.MoldRepairRepository

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TagWriteScreen(onBack: () -> Unit) {
    //api
    val moldRepo = remember { MoldApiRepository() }
    // Stav přepínače MOLD / CARRIER
    var selectedMode by remember { mutableStateOf("MOLD") }

    // Car ComboBox
    var expandedCar by remember { mutableStateOf(false) }
    var selectedCar by remember { mutableStateOf("") }
    var carList by remember { mutableStateOf<List<CarCodeList>>(emptyList()) }

    // Mold ComboBox
    var expandedMold by remember { mutableStateOf(false) }
    var selectedMold by remember { mutableStateOf("") }
    val moldList = listOf("M88150N7001-08", "M88250N7001-08", "M88300N7010-09")

    // Type & Side text fields
    var type by remember { mutableStateOf("") }
    var side by remember { mutableStateOf("") }

    // Stav načítání
    var loading by remember { mutableStateOf(true) }
    var error by remember { mutableStateOf<String?>(null) }

    // Načtení dat z API
    LaunchedEffect(Unit) {
        loading = true
        try {
            carList = moldRepo.getAllCars()
        } catch (e: Exception) {
            error = e.localizedMessage
//            Log.d("Error", error)
        } finally {
            loading = false
        }
    }


    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFECECEC))
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "RF-Tag Write",
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
            // Přepínací tlačítka MOLD / CARRIER
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                ModeButton(
                    text = "MOLD",
                    selected = selectedMode == "MOLD",
                    onClick = { selectedMode = "MOLD" }
                )
                ModeButton(
                    text = "CARRIER",
                    selected = selectedMode == "CARRIER",
                    onClick = { selectedMode = "CARRIER" }
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Car ComboBox
            ExposedDropdownMenuBox(
                expanded = expandedCar,
                onExpandedChange = { expandedCar = !expandedCar }
            ) {
                OutlinedTextField(
                    value = selectedCar,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Car") },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedCar)
                    },
                    modifier = Modifier
                        .menuAnchor()
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                )

                ExposedDropdownMenu(
                    expanded = expandedCar,
                    onDismissRequest = { expandedCar = false }
                ) {
                    carList.forEach { car ->
                        DropdownMenuItem(
                            text = { Text("${car.caR_NAME}") },
                            onClick = {
                                selectedCar = "${car.caR_CODE}"
                                expandedCar = false
                            }
                        )
                    }
                }
            }

            // Mold ComboBox
            ExposedDropdownMenuBox(
                expanded = expandedMold,
                onExpandedChange = { expandedMold = !expandedMold }
            ) {
                OutlinedTextField(
                    value = selectedMold,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Mold") },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedMold)
                    },
                    modifier = Modifier
                        .menuAnchor()
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                )

                ExposedDropdownMenu(
                    expanded = expandedMold,
                    onDismissRequest = { expandedMold = false }
                ) {
                    moldList.forEach { mold ->
                        DropdownMenuItem(
                            text = { Text(mold) },
                            onClick = {
                                selectedMold = mold
                                expandedMold = false
                            }
                        )
                    }
                }
            }

            // Type & Side
            OutlinedTextField(
                value = type,
                onValueChange = { type = it },
                label = { Text("Type") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            )

            OutlinedTextField(
                value = side,
                onValueChange = { side = it },
                label = { Text("Side") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Write Button
            Button(
                onClick = {
                    // TODO: API call to write RF tag
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(55.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF1565C0),
                    contentColor = Color.White
                )
            ) {
                Text(
                    text = "WRITE",
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Read Button (spodní)
            Button(
                onClick = {
                    // TODO: API call to read tag
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(55.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFFFB300),
                    contentColor = Color.Black
                )
            ) {
                Text(
                    text = "READ",
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

@Composable
fun ModeButton(text: String, selected: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) Color(0xFFFFB300) else Color(0xFFBDBDBD),
            contentColor = Color.Black
        ),
        modifier = Modifier
            .height(45.dp)
    ) {
        Text(
            text = text,
            fontWeight = FontWeight.Bold,
            fontSize = 16.sp
        )
    }
}