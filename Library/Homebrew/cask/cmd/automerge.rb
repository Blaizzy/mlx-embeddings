# frozen_string_literal: true

require "cask/cmd/abstract_internal_command"
require "tap"
require "utils/formatter"
require "utils/github"

module Cask
  class Cmd
    class Automerge < AbstractInternalCommand
      OFFICIAL_CASK_TAPS = [
        "homebrew/cask",
        "homebrew/cask-drivers",
        "homebrew/cask-eid",
        "homebrew/cask-fonts",
        "homebrew/cask-versions",
      ].freeze

      def run
        taps = OFFICIAL_CASK_TAPS.map(&Tap.public_method(:fetch))

        access = taps.all? { |tap| GitHub.write_access?(tap.full_name) }
        raise "This command may only be run by Homebrew maintainers." unless access

        Homebrew.install_gem! "git_diff"
        require "git_diff"

        failed = []

        taps.each do |tap|
          open_pull_requests = GitHub.pull_requests(tap.full_name, state: :open, base: "master")

          open_pull_requests.each do |pr|
            next unless passed_ci(pr)
            next unless check_diff(pr)

            number = pr["number"]
            sha = pr.dig("head", "sha")

            print "#{Formatter.url(pr["html_url"])} "

            retried = false

            begin
              GitHub.merge_pull_request(
                tap.full_name,
                number: number, sha: sha,
                merge_method: :squash,
                commit_message: "Squashed and auto-merged via `brew cask automerge`."
              )
              puts "#{Tty.bold}#{Formatter.success("✔")}#{Tty.reset}"
            rescue
              unless retried
                retried = true
                sleep 5
                retry
              end

              puts "#{Tty.bold}#{Formatter.error("✘")}#{Tty.reset}"
              failed << pr["html_url"]
            end
          end
        end

        return if failed.empty?

        $stderr.puts
        raise CaskError, "Failed merging the following PRs:\n#{failed.join("\n")}"
      end

      def passed_ci(pr)
        statuses = GitHub.open_api(pr["statuses_url"])

        latest_pr_status = statuses.select { |status| status["context"] == "continuous-integration/travis-ci/pr" }
                                   .max_by { |status| Time.parse(status["updated_at"]) }

        latest_pr_status&.fetch("state") == "success"
      end

      def check_diff(pr)
        diff_url = pr["diff_url"]

        output, _, status = curl_output("--location", diff_url)

        return false unless status.success?

        diff = GitDiff.from_string(output)

        diff_is_single_cask(diff) && diff_only_version_or_checksum_changed(diff)
      end

      def diff_is_single_cask(diff)
        return false unless diff.files.count == 1

        file = diff.files.first
        return false unless file.a_path == file.b_path

        file.a_path.match?(%r{\ACasks/[^/]+\.rb\Z})
      end

      def diff_only_version_or_checksum_changed(diff)
        lines = diff.files.flat_map(&:hunks).flat_map(&:lines)

        additions = lines.select(&:addition?)
        deletions = lines.select(&:deletion?)
        changed_lines = deletions + additions

        return false if additions.count != deletions.count
        return false if additions.count > 2

        changed_lines.all? { |line| diff_line_is_version(line.to_s) || diff_line_is_sha256(line.to_s) }
      end

      def diff_line_is_sha256(line)
        line.match?(/\A[+-]\s*sha256 '[0-9a-f]{64}'\Z/)
      end

      def diff_line_is_version(line)
        line.match?(/\A[+-]\s*version '[^']+'\Z/)
      end

      def self.help
        "automatically merge “simple” Cask pull requests"
      end
    end
  end
end
