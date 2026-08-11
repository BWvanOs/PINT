def _normalize_channel_name(value: object) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .split()
    )


def _make_unique_channel_names(
    channel_names: list[object],
    channel_labels: list[object],
) -> list[str]:
    """
    Build readable, unique PINT channel names.

    Rules:
    - Prefer biological target labels, e.g. "CD3".
    - Fall back to metal/isotope names if no label exists.
    - Duplicate detection is case-insensitive.
    - If duplicate target labels occur, append the metal/isotope:
          CD3 [Nd142]
          CD3 [Er170]
    - If duplicates still remain, append _2, _3, etc.
    """

    nChannels = max(
        len(channel_names),
        len(channel_labels),
    )

    metals = [
        _normalize_channel_name(
            channel_names[i]
            if i < len(channel_names)
            else ""
        )
        for i in range(nChannels)
    ]

    labels = [
        _normalize_channel_name(
            channel_labels[i]
            if i < len(channel_labels)
            else ""
        )
        for i in range(nChannels)
    ]

    baseNames = []

    for i in range(nChannels):
        if labels[i]:
            baseName = labels[i]
        elif metals[i]:
            baseName = metals[i]
        else:
            baseName = f"Channel{i + 1}"

        baseNames.append(baseName)

    # Count duplicates case-insensitively
    counts = {}

    for name in baseNames:
        key = name.casefold()
        counts[key] = counts.get(key, 0) + 1

    output = []
    used = set()

    for i, baseName in enumerate(baseNames):
        key = baseName.casefold()

        # Add metal/isotope when biological label is duplicated
        if counts[key] > 1 and metals[i]:
            candidate = f"{baseName} [{metals[i]}]"
        else:
            candidate = baseName

        uniqueCandidate = candidate
        suffix = 2

        # Final collision protection is also case-insensitive
        while uniqueCandidate.casefold() in used:
            uniqueCandidate = f"{candidate}_{suffix}"
            suffix += 1

        output.append(uniqueCandidate)
        used.add(uniqueCandidate.casefold())

    return output