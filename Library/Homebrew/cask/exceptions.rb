# frozen_string_literal: true

module Cask
  # General cask error.
  #
  # @api private
  class CaskError < RuntimeError; end

  # Cask error containing multiple other errors.
  #
  # @api private
  class MultipleCaskErrors < CaskError
    def initialize(errors)
      super()

      @errors = errors
    end

    def to_s
      <<~EOS
        Problems with multiple casks:
        #{@errors.map(&:to_s).join("\n")}
      EOS
    end
  end

  # Abstract cask error containing a cask token.
  #
  # @api private
  class AbstractCaskErrorWithToken < CaskError
    attr_reader :token, :reason

    def initialize(token, reason = nil)
      super()

      @token = token
      @reason = reason.to_s
    end
  end

  # Error when a cask is not installed.
  #
  # @api private
  class CaskNotInstalledError < AbstractCaskErrorWithToken
    def to_s
      "Cask '#{token}' is not installed."
    end
  end

  # Error when a cask conflicts with another cask.
  #
  # @api private
  class CaskConflictError < AbstractCaskErrorWithToken
    attr_reader :conflicting_cask

    def initialize(token, conflicting_cask)
      super(token)
      @conflicting_cask = conflicting_cask
    end

    def to_s
      "Cask '#{token}' conflicts with '#{conflicting_cask}'."
    end
  end

  # Error when a cask is not available.
  #
  # @api private
  class CaskUnavailableError < AbstractCaskErrorWithToken
    def to_s
      "Cask '#{token}' is unavailable#{reason.empty? ? "." : ": #{reason}"}"
    end
  end

  # Error when a cask is unreadable.
  #
  # @api private
  class CaskUnreadableError < CaskUnavailableError
    def to_s
      "Cask '#{token}' is unreadable#{reason.empty? ? "." : ": #{reason}"}"
    end
  end

  # Error when a cask already exists.
  #
  # @api private
  class CaskAlreadyCreatedError < AbstractCaskErrorWithToken
    def to_s
      %Q(Cask '#{token}' already exists. Run #{Formatter.identifier("brew cask edit #{token}")} to edit it.)
    end
  end

  # Error when a cask is already installed.
  #
  # @api private
  class CaskAlreadyInstalledError < AbstractCaskErrorWithToken
    def to_s
      <<~EOS
        Cask '#{token}' is already installed.

        To re-install #{token}, run:
          #{Formatter.identifier("brew reinstall #{token}")}
      EOS
    end
  end

  # Error when a cask depends on X11.
  #
  # @api private
  class CaskX11DependencyError < AbstractCaskErrorWithToken
    def to_s
      <<~EOS
        Cask '#{token}' requires XQuartz/X11, which can be installed using Homebrew Cask by running:
          #{Formatter.identifier("brew cask install xquartz")}

        or manually, by downloading the package from:
          #{Formatter.url("https://www.xquartz.org/")}
      EOS
    end
  end

  # Error when there is a cyclic cask dependency.
  #
  # @api private
  class CaskCyclicDependencyError < AbstractCaskErrorWithToken
    def to_s
      "Cask '#{token}' includes cyclic dependencies on other Casks#{reason.empty? ? "." : ": #{reason}"}"
    end
  end

  # Error when a cask depends on itself.
  #
  # @api private
  class CaskSelfReferencingDependencyError < CaskCyclicDependencyError
    def to_s
      "Cask '#{token}' depends on itself."
    end
  end

  # Error when no cask is specified.
  #
  # @api private
  class CaskUnspecifiedError < CaskError
    def to_s
      "This command requires a Cask token."
    end
  end

  # Error when a cask is invalid.
  #
  # @api private
  class CaskInvalidError < AbstractCaskErrorWithToken
    def to_s
      "Cask '#{token}' definition is invalid#{reason.empty? ? "." : ": #{reason}"}"
    end
  end

  # Error when a cask token does not match the file name.
  #
  # @api private
  class CaskTokenMismatchError < CaskInvalidError
    def initialize(token, header_token)
      super(token, "Token '#{header_token}' in header line does not match the file name.")
    end
  end

  # Error with a cask's checksum.
  #
  # @api private
  class CaskSha256Error < AbstractCaskErrorWithToken
    attr_reader :expected, :actual

    def initialize(token, expected = nil, actual = nil)
      super(token)
      @expected = expected
      @actual = actual
    end
  end

  # Error when a cask's checksum is missing.
  #
  # @api private
  class CaskSha256MissingError < CaskSha256Error
    def to_s
      <<~EOS
        Cask '#{token}' requires a checksum:
          #{Formatter.identifier("sha256 \"#{actual}\"")}
      EOS
    end
  end

  # Error when a cask's checksum does not match.
  #
  # @api private
  class CaskSha256MismatchError < CaskSha256Error
    attr_reader :path

    def initialize(token, expected, actual, path)
      super(token, expected, actual)
      @path = path
    end

    def to_s
      <<~EOS
        Checksum for Cask '#{token}' does not match.
        Expected: #{Formatter.success(expected.to_s)}
          Actual: #{Formatter.error(actual.to_s)}
            File: #{path}
        To retry an incomplete download, remove the file above.
        If the issue persists, visit:
          #{Formatter.url("https://github.com/Homebrew/homebrew-cask/blob/HEAD/doc/reporting_bugs/checksum_does_not_match_error.md")}
      EOS
    end
  end

  # Error when a cask has no checksum and the `--require-sha` flag is passed.
  #
  # @api private
  class CaskNoShasumError < CaskSha256Error
    def to_s
      <<~EOS
        Cask '#{token}' does not have a sha256 checksum defined and was not installed.
        This means you have the #{Formatter.identifier("--require-sha")} option set, perhaps in your HOMEBREW_CASK_OPTS.
      EOS
    end
  end

  # Error during quarantining of a file.
  #
  # @api private
  class CaskQuarantineError < CaskError
    attr_reader :path, :reason

    def initialize(path, reason)
      super()

      @path = path
      @reason = reason
    end

    def to_s
      s = +"Failed to quarantine #{path}."

      unless reason.empty?
        s << " Here's the reason:\n"
        s << Formatter.error(reason)
        s << "\n" unless reason.end_with?("\n")
      end

      s.freeze
    end
  end

  # Error while propagating quarantine information to subdirectories.
  #
  # @api private
  class CaskQuarantinePropagationError < CaskQuarantineError
    def to_s
      s = +"Failed to quarantine one or more files within #{path}."

      unless reason.empty?
        s << " Here's the reason:\n"
        s << Formatter.error(reason)
        s << "\n" unless reason.end_with?("\n")
      end

      s.freeze
    end
  end

  # Error while removing quarantine information.
  #
  # @api private
  class CaskQuarantineReleaseError < CaskQuarantineError
    def to_s
      s = +"Failed to release #{path} from quarantine."

      unless reason.empty?
        s << " Here's the reason:\n"
        s << Formatter.error(reason)
        s << "\n" unless reason.end_with?("\n")
      end

      s.freeze
    end
  end
end
