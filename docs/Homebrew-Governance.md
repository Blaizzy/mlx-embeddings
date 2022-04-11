# Homebrew Governance

## 1. Definitions

- PLC: Project Leadership Committee
- TSC: Technical Steering Committee
- AGM: Annual General Meeting
- An ordinary resolution requires a majority of the votes cast.
- A special resolution requires a two-thirds supermajority of the votes cast.

## 2. Members

1. New members (unless nominated as maintainers, see below) will be admitted by an ordinary resolution of the PLC and added to the Homebrew organisation on GitHub.

2. Members may vote in all general elections and resolutions, hold office for Homebrew, and participate in all other membership functions.

3. Members are expected to remain active within Homebrew, and are required to affirm their continued interest in Homebrew membership annually.

4. A member may be removed from Homebrew by an ordinary resolution of the PLC. A removed member may be reinstated by the usual admission process.

5. All members will follow the [Homebrew Code of Conduct](https://github.com/Homebrew/.github/blob/HEAD/CODE_OF_CONDUCT.md#code-of-conduct). Changes to the code of conduct must be approved by the PLC.

6. Members should abstain from voting when they have a conflict of interest not shared by other members. No one may be compelled to abstain from voting.

## 3. General Meetings of Members

1. A general meeting of the members may be called by either an ordinary resolution of the PLC or a majority of the entire membership. The membership must be given at least three weeks notice of a general meeting.

2. The quorum to vote on resolutions and elections at a general meeting is 3 voting members or 10% of the voting members, whichever is greater.

3. Homebrew members will meet at the annual general meeting (AGM) in a manner determined by the PLC.

4. General elections will be held at the AGM.

5. The PLC will announce candidates and proposals three weeks prior to the election date.

6. Members may cast a vote any time up to three weeks prior to the election date.

### 3.1. Amendments to these bylaws

1. These bylaws may be amended by a special resolution at a general meeting of the members.

2. Any member may propose an amendment via pull request on GitHub against this document.

3. Members shall vote on any amendments by approving or requesting changes on the GitHub pull request. Voting will close three weeks after an amendment is proposed, and all votes tallied.

4. Any approved amendments will take effect three weeks after the close of voting.

## 4. Project Leadership Committee

1. The financial administration of Homebrew, organisation of the AGM, enforcement of the code of conduct and removal of members are performed by the PLC. The PLC will represent Homebrew in all dealings with Open Collective.

2. The PLC consists of five members including the Project Leader. Committee members are elected by Homebrew members in a [Meek Single Transferable Vote](https://en.wikipedia.org/wiki/Counting_single_transferable_votes#Meek) election using the Droop quota. Each PLC member will serve a term of two years or until the member's successor is elected. Any sudden vacancy in the PLC will be filled by the usual procedure for electing PLC members at the next general meeting, typically the next AGM.

3. Three weeks prior to the AGM, the PLC will nominate at least one candidate for each upcoming vacant seat. Any member may also nominate any other member as a candidate for the PLC at this time.

4. The PLC will deliver a report of their activities, decisions, and financial statements to the Homebrew members at the AGM.

5. No more than two employees of the same employer may serve on the PLC.

6. A member of the PLC may be removed from the PLC by a special resolution of the membership.

## 5. Meetings of the Project Leadership Committee

1. A synchronous meeting of the PLC may be called by any two of its members with at least three weeks notice, unless all PLC members agree to a shorter notice period.

2. The quorum to vote on resolutions at a synchronous meeting of the PLC is a majority of its members. In a Slack vote, there a time limit instead of quorum: it will take effect after a week, assuming vote passes.

3. A majority of the entire membership of the PLC is required to pass an ordinary resolution.

4. The PLC will meet synchronously and annually to review the status of all members and remove members who did not vote in the AGM and then did not re-affirm a commitment to Homebrew. Voting in the AGM confirms that a member wishes to remain active with the project. After the AGM, the PLC will ask the members who did not vote whether they wish to remain active with the project. The PLC removes any members who don't respond to this second request after three weeks.

5. The PLC will appoint the members of the TSC.

6. Any member may refer any financial questions, AGM questions or code of conduct violations to the PLC. All technical matters should instead be referred to the Project Leader and technical disputes to the TSC. Members will make a good faith effort to resolve any disputes with compromise prior to referral to the PLC, Project Leader or TSC.

7. The PLC may synchronously meet by any mutually agreeable means, such as text chat, voice or video call, and in person. Members of the PLC must meet synchronously at least once per quarter. Members of the PLC must meet by synchronous video call or in person at least once per year.

## 6. Project Leader

1. The Project Leader will represent Homebrew publicly, manage all day-to-day technical decisions, and resolve disputes related to the operation of Homebrew between maintainers, members, other contributors, and users.

2. The Project Leader will be elected annually by Homebrew members in a [Schulze Condorcet method](https://en.wikipedia.org/wiki/Schulze_method) (aka 'beatpath') election. The PLC will nominate at least one candidate for Project Leader. Any member may nominate a candidate, or self-nominate. Nominations must be announced to the membership three weeks before the AGM.

3. Any vacancy of the Project Leader will be filled by appointment of the PLC.

4. The Project Leader's seat on the PLC is non-voting, unless a tie-breaker vote is required.

5. A technical decision of the Project Leader may be overruled by an ordinary resolution of the TSC.

6. A non-technical decision of the Project Leader may be overruled by an ordinary resolution of the PLC.

7. The Project Leader may be removed from the position by a special resolution of the membership.

8. The Project Leader must be included in all PLC communications with or about Open Collective and in all communications related to joint responsibilities.

9. The Project Leader must be a maintainer, not just a member.

## 7. Technical Steering Committee

1. The TSC has the authority to decide on any technical disputes between any maintainer and the Project Leader. Disputes not involving the Project Leader should be addressed through the Project Leader.

2. The PLC will appoint between three and five maintainers to be members of the TSC. Voting PLC members should not be any of these appointees. Appointed TSC members will serve a term of one year or until the member's successor is appointed.

3. Any member may refer any technical question or dispute to the TSC. Members will make a good faith effort to resolve any disputes with compromise prior to referral to the TSC.

4. No more than two employees of the same employer may serve on the TSC.

5. A member of the TSC, except the Project Leader, may be removed from the TSC by an ordinary resolution of the PLC.

## 8. Maintainers

1. All maintainers are automatically members. Some, not all, members are maintainers.

2. Maintainers are members with commit/write-access to at least one of: Homebrew/brew, Homebrew/homebrew-core, Homebrew/homebrew-cask.

3. New maintainers can be nominated by any existing maintainer. They require one approval from the Project Leader or a member of the TSC and no opposition from these people in a 24 hour "working" period (i.e. 7pm on a Friday waits until 7pm on a Monday). If there is opposition, it will be subject to a TSC Slack vote with a majority taking effect after a week.

4. Maintainers' commit access will be reviewed yearly by the Project Leader before the AGM and removed from those who have not committed enough or not performing maintainer responsibilities (e.g. reviewing the PRs of other maintainers and contributors rather than just merging their own PRs) in a given repository. If they are not active enough on any of Homebrew/brew, Homebrew/homebrew-core, Homebrew/homebrew-cask: they will be asked to step down as a maintainer but can remain as a member. If a TSC member believes that a maintainer's removal is unwarranted, they can call a TSC vote on whether to remove the member.

5. Members of the PLC do not have to be maintainers but will have implicit write access to their repositories anyway as "Owners" of the GitHub organisation. If they are not maintainers, they should not make use of this write access.
