# typed: true
# frozen_string_literal: true

module Utils
  # Helper functions for interacting with tar files.
  #
  # @api private
  module Tar
    class << self
      TAR_FILE_EXTENSIONS = %w[.tar .tb2 .tbz .tbz2 .tgz .tlz .txz .tZ].freeze

      def available?
        executable.present?
      end

      def executable
        return @executable if defined?(@executable)

        gnu_tar_gtar_path = HOMEBREW_PREFIX/"opt/gnu-tar/bin/gtar"
        gnu_tar_gtar = gnu_tar_gtar_path if gnu_tar_gtar_path.executable?
        @executable = which("gtar") || gnu_tar_gtar || which("tar")
      end

      def validate_file(path)
        return unless available?

        path = Pathname.new(path)
        return unless TAR_FILE_EXTENSIONS.include? path.extname
        return if Utils.popen_read(executable, "-tf", path).match?(%r{/.*\.})

        odie "#{path} is not a valid tar file!"
      end

      def clear_executable_cache
        remove_instance_variable(:@executable) if defined?(@executable)
      end
    end
  end
end
