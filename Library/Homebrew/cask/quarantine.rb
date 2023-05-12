# typed: true
# frozen_string_literal: true

require "development_tools"
require "cask/exceptions"

module Cask
  # Helper module for quarantining files.
  #
  # @api private
  module Quarantine
    QUARANTINE_ATTRIBUTE = "com.apple.quarantine"

    QUARANTINE_SCRIPT = (HOMEBREW_LIBRARY_PATH/"cask/utils/quarantine.swift").freeze
    COPY_XATTRS_SCRIPT = (HOMEBREW_LIBRARY_PATH/"cask/utils/copy-xattrs.swift").freeze

    def self.swift
      @swift ||= DevelopmentTools.locate("swift")
    end
    private_class_method :swift

    def self.xattr
      @xattr ||= DevelopmentTools.locate("xattr")
    end
    private_class_method :xattr

    def self.swift_target_args
      ["-target", "#{Hardware::CPU.arch}-apple-macosx#{MacOS.version}"]
    end
    private_class_method :swift_target_args

    sig { returns(Symbol) }
    def self.check_quarantine_support
      odebug "Checking quarantine support"

      if !system_command(xattr, args: ["-h"], print_stderr: false).success?
        odebug "There's no working version of `xattr` on this system."
        :xattr_broken
      elsif swift.nil?
        odebug "Swift is not available on this system."
        :no_swift
      else
        api_check = system_command(swift,
                                   args:         [*swift_target_args, QUARANTINE_SCRIPT],
                                   print_stderr: false)

        case api_check.exit_status
        when 2
          odebug "Quarantine is available."
          :quarantine_available
        else
          odebug "Unknown support status"
          :unknown
        end
      end
    end

    def self.available?
      @status ||= check_quarantine_support

      @status == :quarantine_available
    end

    def self.detect(file)
      return if file.nil?

      odebug "Verifying Gatekeeper status of #{file}"

      quarantine_status = !status(file).empty?

      odebug "#{file} is #{quarantine_status ? "quarantined" : "not quarantined"}"

      quarantine_status
    end

    def self.status(file)
      system_command(xattr,
                     args:         ["-p", QUARANTINE_ATTRIBUTE, file],
                     print_stderr: false).stdout.rstrip
    end

    def self.toggle_no_translocation_bit(attribute)
      fields = attribute.split(";")

      # Fields: status, epoch, download agent, event ID
      # Let's toggle the app translocation bit, bit 8
      # http://www.openradar.me/radar?id=5022734169931776

      fields[0] = (fields[0].to_i(16) | 0x0100).to_s(16).rjust(4, "0")

      fields.join(";")
    end

    def self.release!(download_path: nil)
      return unless detect(download_path)

      odebug "Releasing #{download_path} from quarantine"

      quarantiner = system_command(xattr,
                                   args:         [
                                     "-d",
                                     QUARANTINE_ATTRIBUTE,
                                     download_path,
                                   ],
                                   print_stderr: false)

      return if quarantiner.success?

      raise CaskQuarantineReleaseError.new(download_path, quarantiner.stderr)
    end

    def self.cask!(cask: nil, download_path: nil, action: true)
      return if cask.nil? || download_path.nil?

      return if detect(download_path)

      odebug "Quarantining #{download_path}"

      quarantiner = system_command(swift,
                                   args:         [
                                     *swift_target_args,
                                     QUARANTINE_SCRIPT,
                                     download_path,
                                     cask.url.to_s,
                                     cask.homepage.to_s,
                                   ],
                                   print_stderr: false)

      return if quarantiner.success?

      case quarantiner.exit_status
      when 2
        raise CaskQuarantineError.new(download_path, "Insufficient parameters")
      else
        raise CaskQuarantineError.new(download_path, quarantiner.stderr)
      end
    end

    def self.propagate(from: nil, to: nil)
      return if from.nil? || to.nil?

      raise CaskError, "#{from} was not quarantined properly." unless detect(from)

      odebug "Propagating quarantine from #{from} to #{to}"

      quarantine_status = toggle_no_translocation_bit(status(from))

      resolved_paths = Pathname.glob(to/"**/*", File::FNM_DOTMATCH).reject(&:symlink?)

      system_command!("/usr/bin/xargs",
                      args:  [
                        "-0",
                        "--",
                        "/bin/chmod",
                        "-h",
                        "u+w",
                      ],
                      input: resolved_paths.join("\0"))

      quarantiner = system_command("/usr/bin/xargs",
                                   args:         [
                                     "-0",
                                     "--",
                                     xattr,
                                     "-w",
                                     QUARANTINE_ATTRIBUTE,
                                     quarantine_status,
                                   ],
                                   input:        resolved_paths.join("\0"),
                                   print_stderr: false)

      return if quarantiner.success?

      raise CaskQuarantinePropagationError.new(to, quarantiner.stderr)
    end

    sig { params(from: Pathname, to: Pathname).void }
    def self.copy_xattrs(from, to)
      odebug "Copying xattrs from #{from} to #{to}"

      system_command!(
        swift,
        args: [
          *swift_target_args,
          COPY_XATTRS_SCRIPT,
          from,
          to,
        ],
      )
    end
  end
end
