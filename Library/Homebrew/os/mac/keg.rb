# typed: true
# frozen_string_literal: true

class Keg
  def change_dylib_id(id, file)
    return if file.dylib_id == id

    @require_relocation = true
    odebug "Changing dylib ID of #{file}\n  from #{file.dylib_id}\n    to #{id}"
    MachO::Tools.change_dylib_id(file, id, strict: false)
    apply_ad_hoc_signature(file)
  rescue MachO::MachOError
    onoe <<~EOS
      Failed changing dylib ID of #{file}
        from #{file.dylib_id}
          to #{id}
    EOS
    raise
  end

  def change_install_name(old, new, file)
    return if old == new

    @require_relocation = true
    odebug "Changing install name in #{file}\n  from #{old}\n    to #{new}"
    MachO::Tools.change_install_name(file, old, new, strict: false)
    apply_ad_hoc_signature(file)
  rescue MachO::MachOError
    onoe <<~EOS
      Failed changing install name in #{file}
        from #{old}
          to #{new}
    EOS
    raise
  end

  def change_rpath(old, new, file)
    return if old == new

    @require_relocation = true
    odebug "Changing rpath in #{file}\n  from #{old}\n    to #{new}"
    MachO::Tools.change_rpath(file, old, new, strict: false)
    apply_ad_hoc_signature(file)
  rescue MachO::MachOError
    onoe <<~EOS
      Failed changing rpath in #{file}
        from #{old}
          to #{new}
    EOS
    raise
  end

  def delete_rpath(rpath, file)
    odebug "Deleting rpath #{rpath} in #{file}"
    MachO::Tools.delete_rpath(file, rpath, strict: false)
    apply_ad_hoc_signature(file)
  rescue MachO::MachOError
    onoe <<~EOS
      Failed deleting rpath #{rpath} in #{file}
    EOS
    raise
  end

  def apply_ad_hoc_signature(file)
    return if MacOS.version < :big_sur
    return unless Hardware::CPU.arm?

    odebug "Codesigning #{file}"
    # Use quiet_system to squash notifications about resigning binaries
    # which already have valid signatures.
    return if quiet_system("codesign", "--sign", "-", "--force",
                           "--preserve-metadata=entitlements,requirements,flags,runtime",
                           file)

    # If the codesigning fails, it may be a bug in Apple's codesign utility
    # A known workaround is to copy the file to another inode, then move it back
    # erasing the previous file. Then sign again.
    #
    # TODO: remove this once the bug in Apple's codesign utility is fixed
    Dir::Tmpname.create("workaround") do |tmppath|
      FileUtils.cp file, tmppath
      FileUtils.mv tmppath, file, force: true
    end

    # Try signing again
    odebug "Codesigning (2nd try) #{file}"
    result = system_command("codesign", args: [
      "--sign", "-", "--force",
      "--preserve-metadata=entitlements,requirements,flags,runtime",
      file
    ], print_stderr: false)
    return if result.success?

    # If it fails again, error out
    onoe <<~EOS
      Failed applying an ad-hoc signature to #{file}:
      #{result.stderr}
    EOS
  end
end
