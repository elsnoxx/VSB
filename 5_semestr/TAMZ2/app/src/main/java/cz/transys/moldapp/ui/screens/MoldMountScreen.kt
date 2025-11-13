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
fun MoldMountScreen(onBack: () -> Unit) {
    // Carrier ComboBox
    var expandedCarrier by remember { mutableStateOf(false) }
    var selectedCarrier by remember { mutableStateOf("") }
    val carrierList = listOf("#01", "#02", "#03", "#04")

    // Mold #1
    var mold1Type by remember { mutableStateOf("") }
    var mold1Code by remember { mutableStateOf("") }

    // Mold #2
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
        // Title
        Text(
            text = "🟠 MOLD Mount",
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

            // MOLD #1 section
            MoldSection(
                title = "Mold #1",
                titleColor = Color(0xFFFFB300), // oranžová
                type = mold1Type,
                onTypeChange = { mold1Type = it },
                code = mold1Code,
                onCodeChange = { mold1Code = it }
            )

            Spacer(modifier = Modifier.height(12.dp))

            // MOLD #2 section
            MoldSection(
                title = "Mold #2",
                titleColor = Color(0xFFFF8A80), // červenější odstín
                type = mold2Type,
                onTypeChange = { mold2Type = it },
                code = mold2Code,
                onCodeChange = { mold2Code = it }
            )

            Spacer(modifier = Modifier.height(20.dp))

            // MOUNT button
            Button(
                onClick = {
                    // TODO: později API volání
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
                    text = "MOUNT",
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
fun MoldSection(
    title: String,
    titleColor: Color,
    type: String,
    onTypeChange: (String) -> Unit,
    code: String,
    onCodeChange: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(titleColor.copy(alpha = 0.2f))
            .padding(8.dp)
    ) {
        Text(
            text = title,
            fontWeight = FontWeight.Bold,
            fontSize = 18.sp,
            color = titleColor,
            modifier = Modifier.padding(bottom = 4.dp)
        )

        OutlinedTextField(
            value = type,
            onValueChange = onTypeChange,
            label = { Text("Type") },
            placeholder = { Text("např. NX4E FC LH - 08") },
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp)
        )

        OutlinedTextField(
            value = code,
            onValueChange = onCodeChange,
            label = { Text("Mold ID") },
            placeholder = { Text("např. M88150N7001-08") },
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp)
        )
    }
}