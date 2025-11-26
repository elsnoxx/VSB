package cz.transys.moldapp.ui.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.R
import cz.transys.moldapp.ui.apicalls.CarCodeList
import cz.transys.moldapp.ui.apicalls.CarriersList
import cz.transys.moldapp.ui.apicalls.MoldApiRepository
import cz.transys.moldapp.ui.apicalls.MoldRepairRepository
import cz.transys.moldapp.ui.apicalls.MoldsList

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TagWriteScreen(onBack: () -> Unit) {

    val colors = MaterialTheme.colorScheme
    val moldRepo = remember { MoldApiRepository() }

    var selectedMode by remember { mutableStateOf("MOLD") }

    var expandedCar by remember { mutableStateOf(false) }
    var selectedCar by remember { mutableStateOf("") }
    var carList by remember { mutableStateOf<List<CarCodeList>>(emptyList()) }

    var expandedMold by remember { mutableStateOf(false) }
    var selectedMold by remember { mutableStateOf("") }
    val moldList = listOf("M88150N7001-08", "M88250N7001-08", "M88300N7010-09")

    var type by remember { mutableStateOf("FC LH 08") }
    var side by remember { mutableStateOf("LH") }

    var carrierList by remember { mutableStateOf<List<CarriersList>>(emptyList()) }
    var selectedCarrier by remember { mutableStateOf("") }

    var loading by remember { mutableStateOf(true) }
    var error by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        loading = true
        try {
            carList = moldRepo.getAllCars()
            carrierList = moldRepo.getAllCarriers()
        } catch (e: Exception) {
            error = e.localizedMessage
        } finally {
            loading = false
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.tag_write_title),
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = colors.primary,
            modifier = Modifier.padding(bottom = 20.dp)
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.surface)
                .padding(16.dp)
        ) {

            // --- Toggle buttons ---
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
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
            }

            Spacer(modifier = Modifier.height(16.dp))


            // --- Dynamic UI based on mode ---
            when (selectedMode) {
                "MOLD" -> MoldModeSection(
                    repo = moldRepo,
                    selectedCar = selectedCar,
                    onCarChange = { selectedCar = it },
                    selectedMold = selectedMold,
                    onMoldChange = { selectedMold = it },
                    type = type,
                    side = side
                )


                "CARRIER" -> CarrierModeSection(
                    carrierList = carrierList,
                    selectedCarrier = selectedCarrier,
                    onCarrierChange = { selectedCarrier = it }
                )

            }

            Spacer(modifier = Modifier.height(16.dp))

            // --- Buttons ---
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                // WRITE (big)
                Button(
                    onClick = {},
                    modifier = Modifier
                        .weight(2f)
                        .height(55.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colors.primary,
                        contentColor = colors.onPrimary
                    )
                ) {
                    Text(
                        text = stringResource(R.string.write_button),
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                // BACK
                Button(
                    onClick = onBack,
                    modifier = Modifier
                        .weight(1f)
                        .height(55.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colors.secondary,
                        contentColor = colors.onSecondary
                    )
                ) {
                    Text(
                        text = stringResource(R.string.back_button)
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MoldModeSection(
    repo: MoldApiRepository,
    selectedCar: String,
    onCarChange: (String) -> Unit,
    selectedMold: String,
    onMoldChange: (String) -> Unit,
    type: String,
    side: String
) {
    val colors = MaterialTheme.colorScheme

    // Data + loading + error
    var carList by remember { mutableStateOf<List<CarCodeList>>(emptyList()) }
    var moldList by remember { mutableStateOf<List<MoldsList>>(emptyList()) }

    var loadingCars by remember { mutableStateOf(true) }
    var loadingMolds by remember { mutableStateOf(false) }

    var errorMessage by remember { mutableStateOf<String?>(null) }

    // Dropdown states
    var expandedCar by remember { mutableStateOf(false) }
    var expandedMold by remember { mutableStateOf(false) }

    // Načtení seznamu Car (SAFE)
    LaunchedEffect(Unit) {
        loadingCars = true
        try {
            carList = repo.getAllCars()
        } catch (e: Exception) {
            errorMessage = "Failed to load cars: ${e.localizedMessage}"
            carList = emptyList()
        } finally {
            loadingCars = false
        }
    }

    // Po výběru Car → načti Molds
    LaunchedEffect(selectedCar) {
        if (selectedCar.isNotEmpty()) {
            loadingMolds = true
            try {
                moldList = repo.getMoldsByCarCode(selectedCar.lowercase())
            } catch (e: Exception) {
                errorMessage = "Failed to load molds: ${e.localizedMessage}"
                moldList = emptyList()
            } finally {
                loadingMolds = false
            }
        } else {
            moldList = emptyList()
        }
    }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {

        // Zobrazení error hlášky
        errorMessage?.let {
            Text(
                text = it,
                color = Color.Red,
                fontSize = 10.sp,
                fontWeight = FontWeight.Medium
            )
        }

        // CAR COMBOBOX
        ExposedDropdownMenuBox(
            expanded = expandedCar,
            onExpandedChange = { expandedCar = !expandedCar }
        ) {
            OutlinedTextField(
                value = if (loadingCars) "Loading..." else selectedCar,
                onValueChange = {},
                readOnly = true,
                enabled = !loadingCars && carList.isNotEmpty(),
                label = { Text("Car") },
                trailingIcon = {
                    if (!loadingCars) {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedCar)
                    }
                },
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth()
            )

            ExposedDropdownMenu(
                expanded = expandedCar,
                onDismissRequest = { expandedCar = false }
            ) {
                if (carList.isEmpty()) {
                    DropdownMenuItem(
                        text = { Text("No cars available") },
                        onClick = {}
                    )
                } else {
                    carList.forEach { car ->
                        DropdownMenuItem(
                            text = { Text(car.car_name) },
                            onClick = {
                                onCarChange(car.car_code)
                                onMoldChange("")
                                expandedCar = false
                            }
                        )
                    }
                }
            }
        }

        // MOLD COMBOBOX
        ExposedDropdownMenuBox(
            expanded = expandedMold,
            onExpandedChange = { expandedMold = !expandedMold }
        ) {
            OutlinedTextField(
                value = if (loadingMolds) "Loading..." else selectedMold,
                onValueChange = {},
                readOnly = true,
                enabled = selectedCar.isNotEmpty() && moldList.isNotEmpty(),
                label = { Text("Mold") },
                trailingIcon = {
                    if (!loadingMolds && selectedCar.isNotEmpty())
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedMold)
                },
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth()
            )

            ExposedDropdownMenu(
                expanded = expandedMold,
                onDismissRequest = { expandedMold = false }
            ) {
                if (loadingMolds) {
                    DropdownMenuItem(
                        text = { Text("Loading...") },
                        onClick = {}
                    )
                } else if (moldList.isEmpty()) {
                    DropdownMenuItem(
                        text = { Text("No molds available") },
                        onClick = {}
                    )
                } else {
                    moldList.forEach { mold ->
                        DropdownMenuItem(
                            text = { Text("${mold.mold_code} (${mold.mold_side})") },
                            onClick = {
                                onMoldChange(mold.mold_code)
                                expandedMold = false
                            }
                        )
                    }
                }
            }
        }

        // 🟩 TYPE + SIDE (read-only)
        if (selectedMold.isNotEmpty()) {
            ReadOnlyField("Type", type)
            ReadOnlyField("Side", side)
        }
    }
}



@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CarrierModeSection(
    carrierList: List<CarriersList>,
    selectedCarrier: String,
    onCarrierChange: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {

        ExposedDropdownMenuBox(
            expanded = expanded,
            onExpandedChange = { expanded = !expanded }
        ) {
            OutlinedTextField(
                value = selectedCarrier,
                onValueChange = {},
                readOnly = true,
                label = { Text("Carrier") },
                trailingIcon = {
                    ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                },
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth()
            )

            ExposedDropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                carrierList.forEach { carrier ->
                    DropdownMenuItem(
                        text = { Text(carrier.code_value1) },
                        onClick = {
                            onCarrierChange(carrier.code_value2)
                            expanded = false
                        }
                    )
                }
            }
        }

        if (selectedCarrier.isNotEmpty()) {
            ReadOnlyField("Selected Carrier", selectedCarrier)
        }
    }
}





@Composable
fun ModeButton(text: String, selected: Boolean, onClick: () -> Unit) {
    val colors = MaterialTheme.colorScheme

    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) colors.primary else colors.surfaceVariant,
            contentColor = colors.onPrimary
        ),
        modifier = Modifier.height(45.dp)
    ) {
        Text(
            text = text,
            fontWeight = FontWeight.Bold,
            fontSize = 16.sp
        )
    }
}
