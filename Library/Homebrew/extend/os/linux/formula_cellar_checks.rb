# typed: strict
# frozen_string_literal: true

module FormulaCellarChecks
  extend T::Sig
  sig { params(filename: Pathname).returns(T::Boolean) }
  def valid_library_extension?(filename)
    generic_valid_library_extension?(filename) || filename.basename.to_s.include?(".so.")
  end

  sig {params(formula: T.untyped).returns(T.nilable(String))}
  def check_binary_arches(formula)
    return unless formula.prefix.directory?

    keg = Keg.new(formula.prefix)
    mismatches = {}
    keg.binary_executable_or_library_files.each do |file|
      farch = file.arch
      mismatches[file] = farch unless farch == Hardware::CPU.arch
    end
    return if mismatches.empty?

    compatible_universal_binaries, mismatches = mismatches.partition do |file, arch|
      arch == :universal && file.archs.include?(Hardware::CPU.arch)
    end.map(&:to_h) # To prevent transformation into nested arrays

    universal_binaries_expected = if formula.tap.present? && formula.tap.core_tap?
      formula.tap.audit_exception(:universal_binary_allowlist, formula.name)
    else
      true
    end
    return if mismatches.empty? && universal_binaries_expected

    mismatches_expected = formula.tap.blank? ||
                          formula.tap.audit_exception(:mismatched_binary_allowlist, formula.name)
    return if compatible_universal_binaries.empty? && mismatches_expected

    return if universal_binaries_expected && mismatches_expected

    s = ""

    if mismatches.present? && !mismatches_expected
      s += <<~EOS
        Binaries built for a non-native architecture were installed into #{formula}'s prefix.
        The offending files are:
          #{mismatches.map { |m| "#{m.first}\t(#{m.last})" } * "\n  "}
      EOS
    end

    if compatible_universal_binaries.present? && !universal_binaries_expected
      s += <<~EOS
        Unexpected universal binaries were found.
        The offending files are:
          #{compatible_universal_binaries.keys * "\n  "}
      EOS
    end

    s
  end
end
