# typed: true
# frozen_string_literal: true

require "cask/macos"

module Cask
  # Helper class for uninstalling `.pkg` installers.
  #
  # @api private
  class Pkg
    extend T::Sig

    sig { params(regexp: String, command: T.class_of(SystemCommand)).returns(T::Array[Pkg]) }
    def self.all_matching(regexp, command)
      command.run("/usr/sbin/pkgutil", args: ["--pkgs=#{regexp}"]).stdout.split("\n").map do |package_id|
        new(package_id.chomp, command)
      end
    end

    sig { returns(String) }
    attr_reader :package_id

    sig { params(package_id: String, command: T.class_of(SystemCommand)).void }
    def initialize(package_id, command = SystemCommand)
      @package_id = package_id
      @command = command
    end

    sig { void }
    def uninstall
      unless pkgutil_bom_files.empty?
        odebug "Deleting pkg files"
        @command.run!(
          "/usr/bin/xargs",
          args:  ["-0", "--", "/bin/rm", "--"],
          input: pkgutil_bom_files.join("\0"),
          sudo:  true,
        )
      end

      unless pkgutil_bom_specials.empty?
        odebug "Deleting pkg symlinks and special files"
        @command.run!(
          "/usr/bin/xargs",
          args:  ["-0", "--", "/bin/rm", "--"],
          input: pkgutil_bom_specials.join("\0"),
          sudo:  true,
        )
      end

      unless pkgutil_bom_dirs.empty?
        odebug "Deleting pkg directories"
        rmdir(deepest_path_first(pkgutil_bom_dirs))
      end

      rmdir(root) unless MacOS.undeletable?(root)

      forget
    end

    sig { void }
    def forget
      odebug "Unregistering pkg receipt (aka forgetting)"
      @command.run!("/usr/sbin/pkgutil", args: ["--forget", package_id], sudo: true)
    end

    sig { returns(T::Array[Pathname]) }
    def pkgutil_bom_files
      @pkgutil_bom_files ||= pkgutil_bom_all.select(&:file?) - pkgutil_bom_specials
    end

    sig { returns(T::Array[Pathname]) }
    def pkgutil_bom_specials
      @pkgutil_bom_specials ||= pkgutil_bom_all.select(&method(:special?))
    end

    sig { returns(T::Array[Pathname]) }
    def pkgutil_bom_dirs
      @pkgutil_bom_dirs ||= pkgutil_bom_all.select(&:directory?) - pkgutil_bom_specials
    end

    sig { returns(T::Array[Pathname]) }
    def pkgutil_bom_all
      @pkgutil_bom_all ||= @command.run!("/usr/sbin/pkgutil", args: ["--files", package_id])
                                   .stdout
                                   .split("\n")
                                   .map { |path| root.join(path) }
                                   .reject(&MacOS.public_method(:undeletable?))
    end

    sig { returns(Pathname) }
    def root
      @root ||= Pathname.new(info.fetch("volume")).join(info.fetch("install-location"))
    end

    def info
      @info ||= @command.run!("/usr/sbin/pkgutil", args: ["--pkg-info-plist", package_id])
                        .plist
    end

    private

    sig { params(path: Pathname).returns(T::Boolean) }
    def special?(path)
      path.symlink? || path.chardev? || path.blockdev?
    end

    # Helper script to delete empty directories after deleting `.DS_Store` files and broken symlinks.
    # Needed in order to execute all file operations with `sudo`.
    RMDIR_SH = <<~'BASH'
      set -euo pipefail

      for path in "${@}"; do
        if [[ ! -e "${path}" ]]; then
          continue
        fi

        if [[ -e "${path}/.DS_Store" ]]; then
          /bin/rm -f "${path}/.DS_Store"
        fi

        # Some packages leave broken symlinks around; we clean them out before
        # attempting to `rmdir` to prevent extra cruft from accumulating.
        /usr/bin/find "${path}" -mindepth 1 -maxdepth 1 -type l ! -exec /bin/test -e {} \; -delete

        if [[ -L "${path}" ]]; then
          # Delete directory symlink.
          /bin/rm "${path}"
        elif [[ -d "${path}" ]]; then
          # Delete directory if empty.
          /usr/bin/find "${path}" -maxdepth 0 -type d -empty -exec /bin/rmdir {} \;
        else
          # Try `rmdir` anyways to show a proper error.
          /bin/rmdir "${path}"
        fi
      done
    BASH
    private_constant :RMDIR_SH

    sig { params(path: T.any(Pathname, T::Array[Pathname])).void }
    def rmdir(path)
      @command.run!(
        "/usr/bin/xargs",
        args:  ["-0", "--", "/bin/bash", "-c", RMDIR_SH, "--"],
        input: Array(path).join("\0"),
        sudo:  true,
      )
    end

    sig { params(paths: T::Array[Pathname]).returns(T::Array[Pathname]) }
    def deepest_path_first(paths)
      paths.sort_by { |path| -path.to_s.split(File::SEPARATOR).count }
    end
  end
end
