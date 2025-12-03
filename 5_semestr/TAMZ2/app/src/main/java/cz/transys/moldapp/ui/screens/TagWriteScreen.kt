package cz.transys.moldapp.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.R
import cz.transys.moldapp.buisines.apicalls.moldapi.CarCodeList
import cz.transys.moldapp.buisines.apicalls.moldapi.CarriersList
import cz.transys.moldapp.buisines.apicalls.moldapi.MoldApiRepository
import cz.transys.moldapp.buisines.apicalls.moldapi.MoldsList
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TagWriteScreen(onBack: () -> Unit) {

    val colors = MaterialTheme.colorScheme
    val moldRepo = remember { MoldApiRepository() }
    val scope = rememberCoroutineScope()

    var selectedMode by remember { mutableStateOf("MOLD") }
    var selectedCar by remember { mutableStateOf("") }
    var selectedMold by remember { mutableStateOf("") }

    var type by remember { mutableStateOf("FC LH 08") }
    var side by remember { mutableStateOf("LH") }

    var selectedCarrier by remember { mutableStateOf("") }
    var selectedCarrierNo by remember { mutableStateOf("") }

    var showWriteDialog by remember { mutableStateOf(false) }
    var writeStatus by remember { mutableStateOf("idle") }

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

            // Toggle buttons
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


            // Dynamic UI based on mode
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
                    repo = moldRepo,
                    selectedCarrier = selectedCarrier,
                    selectedCarrierNo = selectedCarrierNo,
                    onCarrierChange = { selectedCarrier = it },
                    onCarrierLabel = { selectedCarrierNo = it }
                )

            }

            Spacer(modifier = Modifier.height(16.dp))

            // Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                // WRITE
                Button(
                    onClick = {
                        writeStatus = "writing"
                        showWriteDialog = true

                        scope.launch {
                            delay(1500)

                            val ok = (0..10).random() > 1
                            writeStatus = if (ok) "success" else "error"
                        }
                    },
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

            WriteToRfidDialog(
                show = showWriteDialog,
                status = writeStatus,
                onDismiss = {
                    showWriteDialog = false
                    writeStatus = "idle"
                }
            )

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
    val context = LocalContext.current

    // Data + loading + error
    var carList by remember { mutableStateOf<List<CarCodeList>>(emptyList()) }
    var moldList by remember { mutableStateOf<List<MoldsList>>(emptyList()) }

    var loadingCars by remember { mutableStateOf(true) }
    var loadingMolds by remember { mutableStateOf(false) }

    var errorMessage by remember { mutableStateOf<String?>(null) }

    // Dropdown states
    var expandedCar by remember { mutableStateOf(false) }
    var expandedMold by remember { mutableStateOf(false) }

    // Loading Car list
    LaunchedEffect(Unit) {
        loadingCars = true
        try {
            carList = repo.getAllCars()
        } catch (e: Exception) {
            errorMessage = context.getString(R.string.api_error_full, e.localizedMessage)
            carList = emptyList()
        } finally {
            loadingCars = false
        }
    }

    // After Car list load molds
    LaunchedEffect(selectedCar) {
        if (selectedCar.isNotEmpty()) {
            loadingMolds = true
            try {
                moldList = repo.getMoldsByCarCode(selectedCar.lowercase())
            } catch (e: Exception) {
                errorMessage = context.getString(R.string.api_error_full, e.localizedMessage)
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
                value = if (loadingCars) stringResource(R.string.loading) else selectedCar,
                onValueChange = {},
                readOnly = true,
                enabled = !loadingCars && carList.isNotEmpty(),
                label = { Text(stringResource(R.string.car_label)) },
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
                        text = { Text(stringResource(R.string.no_cars_avaliable)) },
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
                value = if (loadingMolds) stringResource(R.string.loading) else selectedMold,
                onValueChange = {},
                readOnly = true,
                enabled = selectedCar.isNotEmpty() && moldList.isNotEmpty(),
                label = { Text(stringResource(R.string.mold_label)) },
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
                        text = { Text(stringResource(R.string.loading)) },
                        onClick = {}
                    )
                } else if (moldList.isEmpty()) {
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.no_mold_avaliable)) },
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

        // TYPE + SIDE (read-only)
        if (selectedMold.isNotEmpty()) {
            ReadOnlyField(stringResource(R.string.type_label), type)
            ReadOnlyField(stringResource(R.string.side_label), side)
        }
    }
}



@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CarrierModeSection(
    repo: MoldApiRepository,
    selectedCarrier: String,
    selectedCarrierNo: String,
    onCarrierChange: (String) -> Unit,
    onCarrierLabel: (String) -> Unit
) {
    val colors = MaterialTheme.colorScheme
    val context = LocalContext.current
    var expanded by remember { mutableStateOf(false) }
    var carrierList by remember { mutableStateOf<List<CarriersList>>(emptyList()) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var loadingCarrirers by remember { mutableStateOf(true) }

    // Loading Car list
    LaunchedEffect(Unit) {
        loadingCarrirers = true
        try {
            carrierList = repo.getAllCarriers()
        } catch (e: Exception) {
            errorMessage = context.getString(R.string.api_error_full, e.localizedMessage)
            carrierList = emptyList()
        } finally {
            loadingCarrirers = false
        }
    }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {

        ExposedDropdownMenuBox(
            expanded = expanded,
            onExpandedChange = { expanded = !expanded }
        ) {
            OutlinedTextField(
                value = selectedCarrier,
                onValueChange = {},
                readOnly = true,
                label = { Text(stringResource(R.string.carrier_label)) },
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
                            onCarrierChange(carrier.code_value1)
                            onCarrierLabel(carrier.code_value2)
                            expanded = false
                        }
                    )
                }
            }
        }

        if (selectedCarrierNo.isNotEmpty()) {
            ReadOnlyField(stringResource(R.string.selected_carrier), selectedCarrierNo)
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


@Composable
fun WriteToRfidDialog(
    show: Boolean,
    status: String,
    onDismiss: () -> Unit
) {
    if (!show) return

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(stringResource(R.string.dialog_tag_write_title))
        },
        text = {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {

                when (status) {
                    "writing" -> {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(stringResource(R.string.dialog_tag_write_text))
                    }

                    "success" -> {
                        Icon(
                            imageVector = Icons.Default.CheckCircle,
                            contentDescription = null,
                            tint = Color(0xFF43A047),
                            modifier = Modifier.size(48.dp)
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(stringResource(R.string.dialog_tag_write_text_succes))
                    }

                    "error" -> {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = null,
                            tint = Color.Red,
                            modifier = Modifier.size(48.dp)
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(stringResource(R.string.dialog_tag_write_text_fail))
                    }
                }
            }
        },
        confirmButton = {
            if (status != "writing") {
                TextButton(onClick = onDismiss) {
                    Text(stringResource(R.string.dialog_tag_write_button))
                }
            }
        }
    )
}
