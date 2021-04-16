# typed: true
# frozen_string_literal: true

require "system_command"

module UnpackStrategy
  # Strategy for unpacking tar archives.
  class Tar
    extend T::Sig

    include UnpackStrategy
    extend SystemCommand::Mixin

    using Magic

    sig { returns(T::Array[String]) }
    def self.extensions
      [
        ".tar",
        ".tbz", ".tbz2", ".tar.bz2",
        ".tgz", ".tar.gz",
        ".tlzma", ".tar.lzma",
        ".txz", ".tar.xz"
      ]
    end

    def self.can_extract?(path)
      return true if path.magic_number.match?(/\A.{257}ustar/n)

      return false unless [Bzip2, Gzip, Lzip, Xz].any? { |s| s.can_extract?(path) }

      # Check if `tar` can list the contents, then it can also extract it.
      stdout, _, status = system_command("tar", args: ["--list", "--file", path], print_stderr: false)
      status.success? && !stdout.empty?
    end

    private

    sig { override.params(unpack_dir: Pathname, basename: Pathname, verbose: T::Boolean).returns(T.untyped) }
    def extract_to_dir(unpack_dir, basename:, verbose:)
      Dir.mktmpdir do |tmpdir|
        tar_path = path

        if DependencyCollector.tar_needs_xz_dependency? && Xz.can_extract?(path)
          tmpdir = Pathname(tmpdir)
          Xz.new(path).extract(to: tmpdir, verbose: verbose)
          tar_path = tmpdir.children.first
        end

        system_command! "tar",
                        args:    ["--extract", "--no-same-owner",
                                  "--file", tar_path,
                                  "--directory", unpack_dir],
                        verbose: verbose
      end
    end
  end
end
