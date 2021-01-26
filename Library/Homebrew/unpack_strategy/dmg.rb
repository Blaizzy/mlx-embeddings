# typed: false
# frozen_string_literal: true

require "tempfile"

module UnpackStrategy
  # Strategy for unpacking disk images.
  class Dmg
    extend T::Sig

    include UnpackStrategy

    # Helper module for listing the contents of a volume mounted from a disk image.
    module Bom
      DMG_METADATA = Set.new(%w[
                               .background
                               .com.apple.timemachine.donotpresent
                               .com.apple.timemachine.supported
                               .DocumentRevisions-V100
                               .DS_Store
                               .fseventsd
                               .MobileBackups
                               .Spotlight-V100
                               .TemporaryItems
                               .Trashes
                               .VolumeIcon.icns
                             ]).freeze
      private_constant :DMG_METADATA

      refine Pathname do
        extend T::Sig

        # Check if path is considered disk image metadata.
        sig { returns(T::Boolean) }
        def dmg_metadata?
          DMG_METADATA.include?(cleanpath.ascend.to_a.last.to_s)
        end

        # Check if path is a symlink to a system directory (commonly to /Applications).
        sig { returns(T::Boolean) }
        def system_dir_symlink?
          symlink? && MacOS.system_dir?(dirname.join(readlink))
        end

        sig { returns(String) }
        def bom
          tries = 0
          result = loop do
            # rubocop:disable Style/AsciiComments
            # We need to use `find` here instead of Ruby in order to properly handle
            # file names containing special characters, such as “e” + “´” vs. “é”.
            # rubocop:enable Style/AsciiComments
            r = system_command("find", args: [".", "-print0"], chdir: self, print_stderr: false)
            tries += 1

            # Spurious bug on CI, which in most cases can be worked around by retrying.
            break r unless r.stderr.match?(/Interrupted system call/i)

            raise "Command `#{r.command.shelljoin}` was interrupted." if tries >= 3
          end

          odebug "Command `#{result.command.shelljoin}` in '#{self}' took #{tries} tries." if tries > 1

          bom_paths = result.stdout.split("\0")

          raise "BOM for path '#{self}' is empty." if bom_paths.empty?

          bom_paths
            .reject { |path| Pathname(path).dmg_metadata? }
            .reject { |path| (self/path).system_dir_symlink? }
            .join("\n")
        end
      end
    end
    private_constant :Bom

    # Strategy for unpacking a volume mounted from a disk image.
    class Mount
      extend T::Sig

      using Bom
      include UnpackStrategy

      def eject(verbose: false)
        tries ||= 3

        return unless path.exist?

        if tries > 1
          system_command! "diskutil",
                          args:         ["eject", path],
                          print_stderr: false,
                          verbose:      verbose
        else
          system_command! "diskutil",
                          args:         ["unmount", "force", path],
                          print_stderr: false,
                          verbose:      verbose
        end
      rescue ErrorDuringExecution => e
        raise e if (tries -= 1).zero?

        sleep 1
        retry
      end

      private

      sig { override.params(unpack_dir: Pathname, basename: Pathname, verbose: T::Boolean).returns(T.untyped) }
      def extract_to_dir(unpack_dir, basename:, verbose:)
        Tempfile.open(["", ".bom"]) do |bomfile|
          bomfile.close

          Tempfile.open(["", ".list"]) do |filelist|
            filelist.puts(path.bom)
            filelist.close

            system_command! "mkbom",
                            args:    ["-s", "-i", filelist.path, "--", bomfile.path],
                            verbose: verbose
          end

          system_command! "ditto",
                          args:    ["--bom", bomfile.path, "--", path, unpack_dir],
                          verbose: verbose

          FileUtils.chmod "u+w", Pathname.glob(unpack_dir/"**/*", File::FNM_DOTMATCH).reject(&:symlink?)
        end
      end
    end
    private_constant :Mount

    sig { returns(T::Array[String]) }
    def self.extensions
      [".dmg"]
    end

    def self.can_extract?(path)
      stdout, _, status = system_command("hdiutil", args: ["imageinfo", "-format", path], print_stderr: false)
      status.success? && !stdout.empty?
    end

    private

    sig { override.params(unpack_dir: Pathname, basename: Pathname, verbose: T::Boolean).returns(T.untyped) }
    def extract_to_dir(unpack_dir, basename:, verbose:)
      mount(verbose: verbose) do |mounts|
        raise "No mounts found in '#{path}'; perhaps this is a bad disk image?" if mounts.empty?

        mounts.each do |mount|
          mount.extract(to: unpack_dir, verbose: verbose)
        end
      end
    end

    def mount(verbose: false)
      Dir.mktmpdir do |mount_dir|
        mount_dir = Pathname(mount_dir)

        without_eula = system_command(
          "hdiutil",
          args:         [
            "attach", "-plist", "-nobrowse", "-readonly",
            "-mountrandom", mount_dir, path
          ],
          input:        "qn\n",
          print_stderr: false,
          verbose:      verbose,
        )

        # If mounting without agreeing to EULA succeeded, there is none.
        plist = if without_eula.success?
          without_eula.plist
        else
          cdr_path = mount_dir/path.basename.sub_ext(".cdr")

          quiet_flag = "-quiet" unless verbose

          system_command!(
            "hdiutil",
            args:    [
              "convert", *quiet_flag, "-format", "UDTO", "-o", cdr_path, path
            ],
            verbose: verbose,
          )

          with_eula = system_command!(
            "hdiutil",
            args:    [
              "attach", "-plist", "-nobrowse", "-readonly",
              "-mountrandom", mount_dir, cdr_path
            ],
            verbose: verbose,
          )

          if verbose && !(eula_text = without_eula.stdout).empty?
            ohai "Software License Agreement for '#{path}':", eula_text
          end

          with_eula.plist
        end

        mounts = if plist.respond_to?(:fetch)
          plist.fetch("system-entities", [])
               .map { |entity| entity["mount-point"] }
               .compact
               .map { |path| Mount.new(path) }
        else
          []
        end

        begin
          yield mounts
        ensure
          mounts.each do |mount|
            mount.eject(verbose: verbose)
          end
        end
      end
    end
  end
end
